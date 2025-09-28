import os
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MoDRouter(nn.Module):
    """
    MoD Router per recursion step.
    - routing='expert_choice': per-step select top-k tokens by score (independent across steps) => tokens can pass steps 1 and 3 but skip 2.
    - routing='sigmoid_threshold': per-step sigmoid gate in [0,1], select tokens with gate>tau; optional capacity to enforce budget.

    Returns mask m_r in {0,1} for each step r (shape [B,S,1]) and optional aux loss for balance.
    """

    def __init__(
        self,
        hidden: int,
        nr: int,
        routing: str = 'expert_choice',
        k: Optional[int] = None,
        cap_ratio: Optional[List[float]] = None,
        tau: float = 0.5,
        temperature: float = 1.0,
        aux_loss_alpha: float = 0.0,
        use_step_embed: bool = True,
    ):
        super().__init__()
        assert routing in {'expert_choice', 'sigmoid_threshold'}
        self.hidden = hidden
        self.nr = nr
        self.routing = routing
        self.k = k
        self.cap_ratio = cap_ratio
        self.tau = tau
        self.temperature = temperature
        self.aux_loss_alpha = aux_loss_alpha
        self.use_step_embed = use_step_embed

        # lightweight gate net
        self.proj = nn.Linear(hidden, 1)
        if use_step_embed:
            self.step_embed = nn.Embedding(num_embeddings=nr, embedding_dim=1)
        else:
            self.register_parameter('step_embed', None)

    def _capacity_for_step(self, step: int, total_tokens: int) -> int:
        if self.cap_ratio is not None and step < len(self.cap_ratio):
            return max(1, int(round(total_tokens * float(self.cap_ratio[step]))))
        if self.k is not None:
            return min(self.k, total_tokens)
        # default: geometric like [1.0, 2/3, 1/3, ...] truncated
        # normalize so first step ~= total_tokens, later decrease
        ratios = [max(1.0/(i+1), 0.1) for i in range(self.nr)]
        r = ratios[step]
        return max(1, int(round(total_tokens * r)))

    def forward(self, x: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B,S,H]
        Returns: gate g in [0,1] (float), mask m in {0,1} (float), aux_loss (scalar tensor)
        """
        B, S, H = x.shape
        scores = self.proj(x).squeeze(-1)  # [B,S]
        if self.use_step_embed and self.step_embed is not None:
            scores = scores + self.step_embed.weight[step].view(1, 1)
        if self.temperature != 1.0:
            scores = scores / self.temperature

        aux_loss = x.new_zeros(() )
        total_tokens = B * S

        if self.routing == 'expert_choice':
            # select top-k tokens across (B*S) independently per step
            k = self._capacity_for_step(step, total_tokens)
            flat = scores.reshape(-1)  # [B*S]
            topk_vals, topk_idx = torch.topk(flat, k=k, dim=0)
            m = torch.zeros_like(flat, dtype=x.dtype)
            m[topk_idx] = 1.0
            m = m.view(B, S, 1)
            g = torch.sigmoid(scores).unsqueeze(-1)  # gate for info only
            # balance loss: encourage roughly cap ratio usage
            if self.aux_loss_alpha > 0:
                target = torch.tensor(k/total_tokens, dtype=x.dtype, device=x.device)
                used = m.mean()
                aux_loss = self.aux_loss_alpha * (used - target).pow(2)
            return g, m, aux_loss
        else:
            # sigmoid threshold + optional capacity enforcement
            g = torch.sigmoid(scores).unsqueeze(-1)  # [B,S,1]
            m = (g > self.tau).to(dtype=x.dtype)
            # capacity enforcement (keep highest gates)
            cap = self._capacity_for_step(step, total_tokens)
            if m.sum().item() > cap:
                flat_g = g.view(-1)
                topk_vals, topk_idx = torch.topk(flat_g, k=cap, dim=0)
                m = torch.zeros_like(flat_g, dtype=x.dtype)
                m[topk_idx] = 1.0
                m = m.view(B, S, 1)
            if self.aux_loss_alpha > 0:
                target = torch.tensor(cap/total_tokens, dtype=x.dtype, device=x.device)
                used = m.mean()
                aux_loss = self.aux_loss_alpha * (used - target).pow(2)
            return g, m, aux_loss


class RecursionBlock(nn.Module):
    """A tiny MLP block as f(x) for demonstration."""
    def __init__(self, hidden: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden * 4)
        self.fc2 = nn.Linear(hidden * 4, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        return h


class MoRBlock(nn.Module):
    """
    Mixture of Recursions block with per-step router. Parameters of f are shared across steps (Middle-Cycle idea simplified).
    x_{r+1} = x_r + m_r * f(x_r), where m_r in {0,1} is decided independently per step, allowing tokens to visit step 1 and 3 but skip 2.
    """
    def __init__(
        self,
        hidden: int = 128,
        nr: int = 3,
        routing: str = 'expert_choice',
        k: Optional[int] = None,
        cap_ratio: Optional[List[float]] = None,
        tau: float = 0.5,
        temperature: float = 1.0,
        aux_loss_alpha: float = 0.0,
    ):
        super().__init__()
        self.hidden = hidden
        self.nr = nr
        self.f = RecursionBlock(hidden)
        self.router = MoDRouter(hidden, nr, routing=routing, k=k, cap_ratio=cap_ratio, tau=tau,
                                 temperature=temperature, aux_loss_alpha=aux_loss_alpha, use_step_embed=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, H = x.shape
        masks = []
        aux_total = x.new_zeros(())
        h = x
        for r in range(self.nr):
            g_r, m_r, aux = self.router(h, r)  # m_r: [B,S,1]
            y = self.f(h)
            h = h + m_r * y  # bypass when m_r=0
            masks.append(m_r)
            aux_total = aux_total + aux
        masks = torch.stack(masks, dim=1)  # [B, nr, S, 1]
        return h, masks, aux_total


def parse_ratio_list(s: Optional[str], nr: int) -> Optional[List[float]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(',') if p.strip()]
    vals = [float(p) for p in parts]
    if len(vals) < nr:
        # pad with decreasing geometric-like values
        while len(vals) < nr:
            vals.append(max(0.1, vals[-1] * 0.66))
    return vals[:nr]


def main():
    parser = argparse.ArgumentParser(description='MoR (Mixture of Recursions) demo with per-step router (tokens can skip steps).')
    parser.add_argument('--steps', type=int, default=200, help='training steps')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--seq', type=int, default=12)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--nr', type=int, default=3, help='number of recursion steps')
    parser.add_argument('--routing', type=str, default='expert_choice', choices=['expert_choice', 'sigmoid_threshold'])
    parser.add_argument('--k', type=int, default=None, help='capacity per step for expert_choice')
    parser.add_argument('--cap_ratio', type=str, default='', help='comma-separated ratios per step, e.g. 1.0,0.67,0.33')
    parser.add_argument('--tau', type=float, default=0.5, help='threshold for sigmoid routing')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--aux', type=float, default=0.1, help='balance loss weight')
    parser.add_argument('--print_examples', type=int, default=4, help='print token step paths for first N tokens')
    parser.add_argument('--out', type=str, default='mor_token_step_heatmap.png')
    args = parser.parse_args()

    torch.manual_seed(123)
    B, S, H, NR = args.batch, args.seq, args.hidden, args.nr
    x = torch.randn(B, S, H)

    cap_ratio = parse_ratio_list(args.cap_ratio, NR)
    model = MoRBlock(hidden=H, nr=NR, routing=args.routing, k=args.k, cap_ratio=cap_ratio, tau=args.tau,
                     temperature=args.temperature, aux_loss_alpha=args.aux)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    sel_counts = torch.zeros(NR, dtype=torch.long)

    for step in range(args.steps):
        y, masks, aux = model(x)
        # simple self-supervised-style loss: shrink outputs towards zero + aux balance
        loss = (y.pow(2).mean()) + aux
        opt.zero_grad()
        loss.backward()
        opt.step()

        # track selections per step
        with torch.no_grad():
            # masks: [B, NR, S, 1]
            m = masks.squeeze(-1)  # [B, NR, S]
            sel_counts += m.sum(dim=(0, 2)).to(dtype=torch.long)

        if step < 5:
            print(f"step={step} loss={loss.item():.4f} aux={aux.item():.4f}")

    print("Per-step selected token counts:", sel_counts.tolist())

    # Print token paths for first few tokens in the first batch
    with torch.no_grad():
        _, masks, _ = model(x)
        m = masks[0].squeeze(-1)  # [NR, S] for batch 0
        for t in range(min(args.print_examples, S)):
            path = [int(m[r, t].item()) for r in range(NR)]
            print(f"token@pos{t}: passes steps -> {path} (1=pass,0=skip)")

    # Visualization: heatmap of selection probabilities per step (over tokens)
    try:
        import matplotlib.pyplot as plt
        with torch.no_grad():
            _, masks, _ = model(x)
            # average over batch
            m = masks.mean(dim=0).squeeze(-1)  # [NR, S]
        plt.figure(figsize=(8, 4))
        plt.imshow(m.cpu().numpy(), aspect='auto', cmap='Reds', interpolation='nearest')
        plt.colorbar(label='pass ratio')
        plt.yticks(ticks=np.arange(NR), labels=[f'step {i}' for i in range(NR)])
        plt.xlabel('token position')
        plt.title(f'MoR selection heatmap ({args.routing})')
        out_dir = os.path.dirname(__file__)
        out_path = os.path.join(out_dir, args.out)
        plt.tight_layout()
        plt.savefig(out_path)
        print(f'已保存 MoR 选择热图: {out_path}')
    except Exception as e:
        print(f"可视化失败（可忽略）：{e}")


if __name__ == '__main__':
    main()
