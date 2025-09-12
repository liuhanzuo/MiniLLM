import torch
import triton
import triton.language as tl
from ..configs.configs import (
    MATMUL_AUTOTUNE_CONFIGS,
    MATMUL_AUTOTUNE_KEY,
)

def _matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    ### cpu matmul
    return A @ B

@triton.autotune(configs=MATMUL_AUTOTUNE_CONFIGS, key=MATMUL_AUTOTUNE_KEY)
@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :], mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :], mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    
def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Supports A as [M,K] or [B,M,K]; B as [K,N]
    assert A.is_cuda == B.is_cuda, "matmul requires tensors on the same device"
    is_cuda = A.is_cuda
    # Ensure input dtypes match (AMP may make A bf16 and B fp32)
    if A.dtype != B.dtype:
        B = B.to(dtype=A.dtype)
    if A.dim() == 3:
        Bsz, M, K = A.shape
        A2 = A.reshape(Bsz * M, K)
    else:
        Bsz = None
        M, K = A.shape
        A2 = A
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match for matmul"
    if not is_cuda:
        C2 = _matmul(A2, B)
        C = C2 if Bsz is None else C2.view(Bsz, M, N)
        return C

    C2 = torch.empty((A2.shape[0], N), device=A.device, dtype=torch.float32)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    _matmul_kernel[grid](A2, B, C2, M=A2.shape[0], N=N, K=K)
    C2 = C2.to(A.dtype) if C2.dtype != A.dtype else C2
    C = C2 if Bsz is None else C2.view(Bsz, M, N)
    return C
