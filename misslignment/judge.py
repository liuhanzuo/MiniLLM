import argparse
import torch
from MiniLLM.model.model import MiniLLMLM
from MiniLLM.model.LMConfig import LMConfig
from MiniLLM.model.tokenizer_utils import build_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckp', type=str, required=True)
    ap.add_argument('--tokenizer_dir', type=str, default='MiniLLM/model/minillm_tokenizer')
    ap.add_argument('--dim', type=int, default=512)
    ap.add_argument('--n_layers', type=int, default=8)
    ap.add_argument('--n_block', type=int, default=None)
    ap.add_argument('--max_seq_len', type=int, default=512)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    tokenizer = build_tokenizer(args.tokenizer_dir, trust_remote_code=True)
    lm_conf = LMConfig(dim=args.dim, n_layers=args.n_layers, n_block=args.n_block, max_seq_len=args.max_seq_len)
    model = MiniLLMLM(lm_conf)
    state = torch.load(args.ckp, map_location=device)
    state = {k: v for k, v in state.items() if 'mask' not in k}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    print('输入内容，空行退出：')
    while True:
        text = input('> ').strip()
        if not text:
            break
        prompt = tokenizer.bos_token + text
        x = torch.tensor(tokenizer(prompt)['input_ids'], device=device).unsqueeze(0)
        out = model.generate(x, max_new_tokens=args.max_seq_len, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        print(tokenizer.decode(out[0][x.shape[1]:].tolist(), skip_special_tokens=True))


if __name__ == '__main__':
    main()
