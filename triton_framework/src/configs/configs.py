import triton

MATMUL_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),

    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=8, num_stages=2),

    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=2, num_stages=2),
]

MATMUL_AUTOTUNE_KEY = ["M", "N", "K"]

SILU_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 512}, num_warps=2),
    triton.Config({"BLOCK": 2048}, num_warps=8),
]

SILU_AUTOTUNE_KEY = ["size"]
