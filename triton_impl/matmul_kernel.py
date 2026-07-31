import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    a_ptr,b_ptr,c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = offs_am < M
    mask_n = offs_bn < N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    a_offs = offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_offs = offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = (offs_k + k * BLOCK_SIZE_K) < K
        mask_a = mask_k[None, :] & mask_m[:, None]
        mask_b = mask_k[:, None] & mask_n[None, :]
        
        a = tl.load(a_ptr + a_offs,mask=mask_a, other=0.0)
        b = tl.load(b_ptr + b_offs, mask=mask_b, other=0.0)
        
        accumulator += tl.dot(a, b)
        
        a_offs += BLOCK_SIZE_K * stride_ak
        b_offs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    c_offs_m = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    c_offs_n = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N
    mask_c = (c_offs_m[:,None] < M) & (c_offs_n[None, :] < N)
    tl.store(c_ptr + c_offs_m[:, None] * stride_cm + c_offs_n[None,:] * stride_cn, c, mask=mask_c)

@triton.jit
def grouped_matmul_kernel(
    a_ptr,b_ptr,c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    # 沿着M N方向分别有几个block
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    num_groups = GROUP_SIZE_M * grid_n
    group_id = pid // num_groups
    group_size = tl.minimum(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * group_size + (pid % group_size)
    pid_n = (pid % num_groups) // group_size
    
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = offs_am < M
    mask_n = offs_bn < N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    a_offs = offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_offs = offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = (offs_k + k * BLOCK_SIZE_K) < K
        mask_a = mask_k[None, :] & mask_m[:, None]
        mask_b = mask_k[:, None] & mask_n[None, :]
        
        a = tl.load(a_ptr + a_offs,mask=mask_a, other=0.0)
        b = tl.load(b_ptr + b_offs, mask=mask_b, other=0.0)
        
        accumulator += tl.dot(a, b)
        
        a_offs += BLOCK_SIZE_K * stride_ak
        b_offs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    c_offs_m = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    c_offs_n = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N
    mask_c = (c_offs_m[:,None] < M) & (c_offs_n[None, :] < N)
    tl.store(c_ptr + c_offs_m[:, None] * stride_cm + c_offs_n[None,:] * stride_cn, c, mask=mask_c)

def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_K=16,
    )
    return c

def group_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    grouped_matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_K=16,
        GROUP_SIZE_M=4,
    )
    return c

def run_benchmark_fixed():
    # 测试更大规格，让矩阵总量 (256MB) 彻底超过 64MB L2 Cache
    matrix_sizes = [1024, 2048, 4096, 8192, 9216]
    
    print(f"{'Matrix Size':<15} | {'Naïve Matmul (ms)':<20} | {'Grouped Matmul (ms)':<20} | {'Speedup':<10}")
    print("-" * 75)

    for size in matrix_sizes:
        M = N = K = size
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        
        res_naive = matmul(a, b)
        res_grouped = group_matmul(a, b)
        
        # 1. 直接对比两个 Triton Kernel 的结果（确保绝对逻辑一致）
        torch.testing.assert_close(res_grouped, res_naive, atol=1e-3, rtol=1e-3)

        # 2. 测量时间
        ms_naive = triton.testing.do_bench(lambda: matmul(a, b))
        ms_grouped = triton.testing.do_bench(lambda: group_matmul(a, b))
        
        speedup = ms_naive / ms_grouped
        print(f"{size}x{size}x{size:<7} | {ms_naive:<20.4f} | {ms_grouped:<20.4f} | {speedup:.2f}x")

if __name__ == "__main__":
    run_benchmark_fixed()