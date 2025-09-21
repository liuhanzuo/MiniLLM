import argparse
import json
import os
import time
from typing import Dict, Any, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    # 新版 transformers 提供更可靠的检测接口
    from transformers.utils.import_utils import is_flash_attn_2_available as hf_flash2_available
except Exception:
    hf_flash2_available = None


def fmt_mib(x_bytes: float) -> float:
    return round(x_bytes / (1024**2), 2)

def get_peak_reserved_mib_all() -> float:
    if not torch.cuda.is_available():
        return 0.0
    total = 0
    for i in range(torch.cuda.device_count()):
        try:
            total += torch.cuda.max_memory_reserved(i)
        except TypeError:
            with torch.cuda.device(i):
                total += torch.cuda.max_memory_reserved()
    return fmt_mib(total)


def get_peak_allocated_mib_all() -> float:
    if not torch.cuda.is_available():
        return 0.0
    total = 0
    for i in range(torch.cuda.device_count()):
        try:
            total += torch.cuda.max_memory_allocated(i)
        except TypeError:
            with torch.cuda.device(i):
                total += torch.cuda.max_memory_allocated()
    return fmt_mib(total)


def reset_cuda_peak_stats_all():
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(i)
        except TypeError:
            with torch.cuda.device(i):
                torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def detect_flash_attn_available() -> bool:
    # 判定 flash-attn 是否可用
    try: 
        if hf_flash2_available is not None:
            return bool(hf_flash2_available())
    except Exception:
        pass
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def set_attn_backend(model, backend: str):
    """
    backend in {"default", "sdpa", "flash_attn"}
    通过 monkey patch 或配置尽量启用目标注意力实现。
    - sdpa: 使用 PyTorch 2.x 的 scaled_dot_product_attention
    - flash_attn: 仅在可用并且模型支持时启用；否则回退 default
    """
    backend = backend.lower()

    # transformers 近版本可通过 config 与 env 影响；这里尽量用 env+attrs
    if backend == "sdpa":
        os.environ["PYTORCH_USE_CUDA_DSA"] = "1"  # noop 仅示例
        try:
            model.config._attn_implementation = "sdpa"
        except Exception:
            pass
    elif backend == "flash_attn":
        if detect_flash_attn_available():
            try:
                model.config._attn_implementation = "flash_attention_2"
            except Exception:
                pass
        else:
            print("[warn] flash-attn not available; fallback to default")
            try:
                model.config._attn_implementation = "eager"
            except Exception:
                pass
    else:
        # default/eager
        try:
            model.config._attn_implementation = "eager"
        except Exception:
            pass


def generate_once(model, tokenizer, prompt: str, max_new_tokens: int, use_kv_cache: bool, device: torch.device) -> Dict[str, Any]:
    # 清理并准备峰值统计
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        reset_cuda_peak_stats_all()

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_kv_cache

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    with torch.inference_mode():
        # transformers generate 在启用 kv cache 时默认使用 past_key_values
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_kv_cache,
        )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()

    new_tokens = outputs[0].shape[-1] - inputs["input_ids"].shape[-1]
    elapsed = t1 - t0
    toks_per_sec = new_tokens / elapsed if elapsed > 0 else 0.0

    peak_reserved = get_peak_reserved_mib_all()
    peak_alloc = get_peak_allocated_mib_all()

    return {
        "new_tokens": int(new_tokens),
        "elapsed": round(elapsed, 4),
        "toks_per_sec": round(toks_per_sec, 2),
        "peak_reserved_mib": peak_reserved,
        "peak_allocated_mib": peak_alloc,
    }


def choose_dtype(user_dtype: str) -> torch.dtype:
    user_dtype = (user_dtype or "auto").lower()
    if user_dtype == "bf16":
        return torch.bfloat16
    if user_dtype in ("fp16", "float16", "half"):
        return torch.float16
    if user_dtype in ("fp32", "float32"):
        return torch.float32
    # auto
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    return torch.float32


def run_suite_with_model(model, tokenizer, prompt: str, max_new_tokens: int, warmup: int, runs: int, backend: str, use_kv_cache: bool, device: torch.device, baseline_mem: Tuple[float, float]) -> Dict[str, Any]:
    # 切换注意力后端（不重载模型）
    set_attn_backend(model, backend)
    # 预热
    for _ in range(warmup):
        _ = generate_once(model, tokenizer, prompt, max_new_tokens, use_kv_cache, device)
    # 正式多次运行统计
    records: List[Dict[str, Any]] = []
    for _ in range(runs):
        rec = generate_once(model, tokenizer, prompt, max_new_tokens, use_kv_cache, device)
        records.append(rec)
    # 聚合
    avg_elapsed = sum(r["elapsed"] for r in records) / max(1, len(records))
    avg_tps = sum(r["toks_per_sec"] for r in records) / max(1, len(records))
    max_peak_reserved = max(r["peak_reserved_mib"] for r in records)
    max_peak_alloc = max(r["peak_allocated_mib"] for r in records)
    base_reserved, base_alloc = baseline_mem
    delta_reserved = round(max(0.0, max_peak_reserved - base_reserved), 2)
    delta_alloc = round(max(0.0, max_peak_alloc - base_alloc), 2)
    return {
        "backend": backend,
        "use_kv_cache": use_kv_cache,
        "avg_elapsed": round(avg_elapsed, 4),
        "avg_toks_per_sec": round(avg_tps, 2),
        "max_peak_reserved_mib": max_peak_reserved,
        "max_peak_allocated_mib": max_peak_alloc,
        "delta_reserved_mib": delta_reserved,
        "delta_allocated_mib": delta_alloc,
        "runs": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HF model id, e.g., facebook/opt-1.3b")
    parser.add_argument("--prompt", type=str, default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32", "float16", "float32"], help="Model dtype. auto=> bf16 if supported else fp16 (GPU) else fp32")
    parser.add_argument("--auto_recover_fp32", action="store_true", help="遇到浮点异常时自动切换到 fp32 重试（仍在 GPU 上）")
    parser.add_argument("--prompt_tokens", type=int, default=0, help="若>0，则生成一个近似此长度的合成长prompt，覆盖 --prompt")
    parser.add_argument("--only_backends", type=str, default="default,sdpa,flash_attn", help="逗号分隔，可选 default,sdpa,flash_attn")
    parser.add_argument("--tp", type=str, default="auto", choices=["auto", "none", "balanced", "greedy"], help="多GPU策略：auto/balanced 为权重分片；none 仅用 1 卡；greedy 表示每卡一整份模型以提升吞吐（高显存占用）")
    parser.add_argument("--gpu_mem_fraction", type=float, default=0.85, help="为每块GPU预留的内存占比上限，用于 device_map 分片时构造 max_memory")
    parser.add_argument("--concurrency", type=int, default=0, help="greedy 模式下的并发度（默认=GPU 数量）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构造测试后端
    allowed = {s.strip() for s in args.only_backends.split(',') if s.strip()}
    suites = []
    if "default" in allowed:
        suites.append(("default", False))
        suites.append(("default", True))
    if "sdpa" in allowed:
        suites.append(("sdpa", True))
    if "flash_attn" in allowed:
        suites.append(("flash_attn", True))

    # 构造长 prompt（若指定）
    prompt = args.prompt
    if args.prompt_tokens and args.prompt_tokens > 0:
        # 生成一个由简单词重复组成的长提示
        unit = "The quick brown fox jumps over the lazy dog. "
        text = (unit * ((args.prompt_tokens // 9) + 1))[:args.prompt_tokens*2]
        prompt = text

    # 加载一次 tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    dtype = choose_dtype(args.dtype)
    # 自动检测多卡配置
    models = None
    model = None
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if args.tp == "greedy" and n_gpus > 1:
            # 每卡加载一份完整模型（显存占用为 N 倍）
            models = []
            conc = args.concurrency if args.concurrency > 0 else n_gpus
            print(f"[info] Greedy DP mode: replicate model on {n_gpus} GPUs, concurrency={conc}")
            for i in range(n_gpus):
                print(f"  loading replica on cuda:{i} ...")
                m = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    dtype=dtype,
                    device_map={"": i},
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                models.append(m)
        else:
            # 构造 device_map 与 max_memory（分片或单卡）
            device_map = None
            max_memory = None
            if args.tp != "none" and n_gpus > 1:
                device_map = "auto" if args.tp == "auto" else "balanced"
                max_memory = {}
                for i in range(n_gpus):
                    props = torch.cuda.get_device_properties(i)
                    total = props.total_memory
                    allow = int(total * max(0.1, min(args.gpu_mem_fraction, 0.95)))
                    max_memory[i] = allow
                print(f"[info] Detected {n_gpus} GPUs; using tensor-parallel device_map={device_map}, gpu_mem_fraction={args.gpu_mem_fraction}")
            else:
                device_map = {"": 0}
                print(f"[info] Using single GPU (device 0). Set --tp auto/greedy to enable multi-GPU.")

            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=dtype,
                device_map=device_map,
                max_memory=max_memory,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
    else:
        print("[info] CUDA not available; running on CPU.")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    # baseline 内存（加载模型后，未开始生成）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        reset_cuda_peak_stats_all()
        base_reserved = get_peak_reserved_mib_all()
        base_alloc = get_peak_allocated_mib_all()
    else:
        base_reserved = base_alloc = 0.0
    baseline_mem = (base_reserved, base_alloc)

    def run_dp_greedy(models_list, backend: str, use_kv: bool) -> Dict[str, Any]:
        # 为所有副本设置 backend
        for m in models_list:
            set_attn_backend(m, backend)
        # 预热：各卡并发执行一次
        from concurrent.futures import ThreadPoolExecutor
        def run_one(m):
            dev = next(m.parameters()).device
            return generate_once(m, tokenizer, prompt, args.max_new_tokens, use_kv, dev)
        with ThreadPoolExecutor(max_workers=len(models_list)) as ex:
            _ = list(ex.map(run_one, models_list))
        # 正式 runs：统计总吞吐
        records = []
        for _ in range(args.runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=len(models_list)) as ex:
                outs = list(ex.map(run_one, models_list))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            total_tokens = sum(o["new_tokens"] for o in outs)
            elapsed = max(1e-6, t1 - t0)
            agg_tps = total_tokens / elapsed
            peak_reserved = get_peak_reserved_mib_all()
            peak_alloc = get_peak_allocated_mib_all()
            rec = {
                "new_tokens": int(total_tokens),
                "elapsed": round(elapsed, 4),
                "toks_per_sec": round(agg_tps, 2),
                "peak_reserved_mib": peak_reserved,
                "peak_allocated_mib": peak_alloc,
            }
            records.append(rec)
        avg_elapsed = sum(r["elapsed"] for r in records) / max(1, len(records))
        avg_tps = sum(r["toks_per_sec"] for r in records) / max(1, len(records))
        max_peak_reserved = max(r["peak_reserved_mib"] for r in records)
        max_peak_alloc = max(r["peak_allocated_mib"] for r in records)
        delta_reserved = round(max(0.0, max_peak_reserved - baseline_mem[0]), 2)
        delta_alloc = round(max(0.0, max_peak_alloc - baseline_mem[1]), 2)
        return {
            "backend": backend,
            "use_kv_cache": use_kv,
            "replicated": True,
            "avg_elapsed": round(avg_elapsed, 4),
            "avg_toks_per_sec": round(avg_tps, 2),
            "max_peak_reserved_mib": max_peak_reserved,
            "max_peak_allocated_mib": max_peak_alloc,
            "delta_reserved_mib": delta_reserved,
            "delta_allocated_mib": delta_alloc,
            "runs": records,
        }

    all_results: List[Dict[str, Any]] = []
    if models is not None:
        # Greedy DP 路径
        for backend, use_kv in suites:
            print(f"\n=== Running [DP Greedy] backend={backend}, use_kv_cache={use_kv} ===")
            try:
                res = run_dp_greedy(models, backend, use_kv)
                all_results.append(res)
                print(json.dumps(res, indent=2))
            except Exception as e:
                print(f"[skip] DP greedy backend={backend} error: {e}")
    else:
        # 单模型（分片/单卡）路径
        for backend, use_kv in suites:
            print(f"\n=== Running backend={backend}, use_kv_cache={use_kv} ===")
            try:
                res = run_suite_with_model(model, tokenizer, prompt, args.max_new_tokens, args.warmup, args.runs, backend, use_kv, device, baseline_mem)
                res["replicated"] = False
                all_results.append(res)
                print(json.dumps(res, indent=2))
            except RuntimeError as e:
                msg = str(e)
                print(f"[error] backend={backend} RuntimeError: {msg}")
                if args.auto_recover_fp32 and ("Floating point" in msg or "floating point" in msg) and torch.cuda.is_available():
                    try:
                        print("[recover] Switching model to fp32 and retrying on GPU...")
                        model.to(dtype=torch.float32)
                        res = run_suite_with_model(model, tokenizer, prompt, args.max_new_tokens, args.warmup, args.runs, backend, use_kv, device, baseline_mem)
                        res["replicated"] = False
                        all_results.append(res)
                        print(json.dumps(res, indent=2))
                        continue
                    except Exception as e2:
                        print(f"[recover-failed] {e2}")
                print(f"[skip] backend={backend} errored: {e}")
            except Exception as e:
                print(f"[skip] backend={backend} unexpected error: {e}")

    # 保存 JSON/CSV 与一个简单的图
    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # CSV
    csv_path = os.path.join(args.out, "results.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "use_kv_cache", "replicated", "avg_elapsed", "avg_toks_per_sec", "max_peak_reserved_mib", "max_peak_allocated_mib", "delta_reserved_mib", "delta_allocated_mib"])
        for r in all_results:
            writer.writerow([
                r["backend"], r["use_kv_cache"], r.get("replicated", False), r["avg_elapsed"], r["avg_toks_per_sec"], r["max_peak_reserved_mib"], r["max_peak_allocated_mib"], r["delta_reserved_mib"], r["delta_allocated_mib"],
            ])

    # 简单可视化
    try:
        import matplotlib.pyplot as plt
        labels = [f"{r['backend']}{' [DP]' if r.get('replicated', False) else ''}\nKV={r['use_kv_cache']}" for r in all_results]
        tps = [r["avg_toks_per_sec"] for r in all_results]
        mem = [r["delta_reserved_mib"] for r in all_results]
    
        fig, axs = plt.subplots(1, 2, figsize=(10,4))
        axs[0].bar(labels, tps, color="tab:blue")
        axs[0].set_title("Tokens/sec (higher is better)")
        axs[0].set_ylabel("tok/s")
        axs[1].bar(labels, mem, color="tab:orange")
        axs[1].set_title("Extra reserved MiB over baseline (lower is better)")
        axs[1].set_ylabel("MiB")
        plt.tight_layout()
        fig_path = os.path.join(args.out, "results.png")
        plt.savefig(fig_path)
        print(f"Saved plot to {fig_path}")
    except Exception as e:
        print(f"[warn] plotting failed: {e}")

    print(f"Saved json to {json_path}\nSaved csv to {csv_path}")


if __name__ == "__main__":
    main()
