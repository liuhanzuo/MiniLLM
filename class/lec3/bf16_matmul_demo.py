"""
BF16 矩阵乘法精度演示

说明：BF16 与 FP32 具有相同数量的指数位，许多实现会在矩阵乘法时将 BF16 输入上转换为 FP32 进行乘累加，
在得到结果后再下转换回 BF16，可显著提升数值精度（相较于直接在 BF16 中累加）。

对比思路：
- R1 = BF16(A) @ BF16(B) -> FP32：先将输入量化到 BF16，再做矩阵乘（库内部通常 FP32 累加），最后转回 FP32 便于比较；
- R2 = FP32(A @ B) -> BF16 -> FP32：先用原始 FP32 做乘法，再整体量化为 BF16（只在输出端量化）。
注意：由于 R1 在“输入端”就已量化，R1 与 R2 一般不会完全一致（输入量化与输出量化的位置不同）。
为了侧证“是否采用 FP32 累加”，再构造：
- R3 = FP32( BF16(A) ) @ FP32( BF16(B) ) -> BF16 -> FP32：即显式用 FP32 对“已量化的输入”做乘法后再量化。
若底层实现采用 FP32 累加，则 R1 与 R3 应高度一致（通常极小差异，很多平台可逐元素相等）。
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="BF16 matmul precision demo")
    parser.add_argument("--m", type=int, default=256, help="Rows of A and R")
    parser.add_argument("--k", type=int, default=256, help="Cols of A / Rows of B")
    parser.add_argument("--n", type=int, default=256, help="Cols of B and R")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # 生成 FP32 矩阵
    A = torch.randn(args.m, args.k, dtype=torch.float32, device=device)
    B = torch.randn(args.k, args.n, dtype=torch.float32, device=device)

    # 参考：FP32 直接相乘
    AB_fp32 = A @ B  # [m, n], float32

    # 路线1：先降精度到 BF16 再相乘；结果再升回 FP32 便于比较
    A_bf16 = A.to(torch.bfloat16)
    B_bf16 = B.to(torch.bfloat16)
    R1 = (A_bf16 @ B_bf16).to(torch.float32)

    # 路线2：先在 FP32 中相乘，再整体降为 BF16；结果同样升回 FP32 用于比较
    R2 = AB_fp32.to(torch.bfloat16).to(torch.float32)

    # 路线3：先将输入量化为 BF16，再显式用 FP32 做乘法，然后整体降到 BF16；最后升回 FP32 用于比较
    R3 = (A_bf16.to(torch.float32) @ B_bf16.to(torch.float32)).to(torch.bfloat16).to(torch.float32)

    # 误差统计
    diff12 = (R1 - R2).abs()
    diff13 = (R1 - R3).abs()
    max_diff12 = diff12.max().item()
    mean_diff12 = diff12.mean().item()
    eq_ratio12 = (diff12 == 0).float().mean().item()
    exact_equal12 = torch.allclose(R1, R2, atol=0, rtol=0)

    max_diff13 = diff13.max().item()
    mean_diff13 = diff13.mean().item()
    eq_ratio13 = (diff13 == 0).float().mean().item()
    exact_equal13 = torch.allclose(R1, R3, atol=0, rtol=0)

    print("=== BF16 MatMul Precision Demo ===")
    print(f"device        : {device}")
    print(f"shapes        : A=({args.m},{args.k}), B=({args.k},{args.n})")
    print(f"dtypes        : A={A.dtype}, B={B.dtype}, A_bf16={A_bf16.dtype}, B_bf16={B_bf16.dtype}")
    print("Compute paths :")
    print("  R1 = BF16(A) @ BF16(B) -> FP32")
    print("  R2 = FP32(A@B) -> BF16 -> FP32")
    print("  R3 = FP32(BF16(A)) @ FP32(BF16(B)) -> BF16 -> FP32")
    print(f"max |R1-R2|   : {max_diff12:.6e}")
    print(f"mean |R1-R2|  : {mean_diff12:.6e}")
    print(f"eq%  |R1-R2|  : {eq_ratio12*100:.2f}%  (percentage of exactly equal elements)")
    print(f"exact equal   : {exact_equal12}")
    print("-")
    print(f"max |R1-R3|   : {max_diff13:.6e}")
    print(f"mean |R1-R3|  : {mean_diff13:.6e}")
    print(f"eq%  |R1-R3|  : {eq_ratio13*100:.2f}%")
    print(f"exact equal   : {exact_equal13}")

    print("\n解读：R1 与 R2 的差异来自“输入量化 vs 输出量化”的不同位置；")
    print("若底层采用 'BF16 输入 -> FP32 累加 -> BF16 输出'，则 R1 应与 R3 高度一致（许多 GPU/库上可逐元素相等）。")
    print("建议在支持 Tensor Core 的 GPU 上加上 --device cuda 观察 R1≈R3 的一致性更明显。")


if __name__ == "__main__":
    main()
