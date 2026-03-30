"""
This matmul kernel can be a bit confusing but is very crucial to understand

 What you'll learn:
- Automatic performance tuning
- Program re-ordering for improved SRAM hit rate
- Multi-dimensional pointer arithmetic
- High precision data type accumulation
- using the Triton interpreter (kind of)

Recommended order to read the code in:
Step 1 - unit test
Step 2 - wrapper
Step 3 - kernel
Step 4 - benchmark

For matmul of A @ B = C of shapes (M, K) @ (K, N) = (M, N), the following
algorithm is numerically equivalent to what our code will output, but we'll
get to the answer in a different way
for m in range(0, M, BLOCK_SIE_M): # do in parallel
    for n in range(0, N, BLOCK_SIZE_N): # do in parallel
        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
        for k in range(0, K, BLOCK_SIZE_K):
            a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
            b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
            acc += dot(a,b)
        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc

see original
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
"""
import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

######### Step 3 #########

# un-comment this to run a numpy emulation of Triton on CPU & be able to debug with print() statements
# import os

# os.environ["TRITON_INTERPRET"] = "1"

# autotuning is just setting up a bunch of different potential meta-parameters configurations that Triton will automatically
# choose from later based on which one performs best on our specific GPU. Triton will figure out for us which one to use. They're
# all values chosen heuristically, but notice everything is a multiple of 32 in sticking w/ the number of threads in a warp.
autotune_configs = [
    # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3,
    #               num_warps=8),
    # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
    #               num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
    #               num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
    #               num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
    #               num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
    #               num_warps=4),
    # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5,
    #               num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE': 4}, num_stages=5,
                  num_warps=2)
]


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator which consumes
#   1) a list of `triton.Config` objects that define different configs of meta-parameters and compilation options
#   2) an auto-tuning *key* whose change in values will trigger a new evaluation of all the provided configs, meaning
#       that any time either M, N, or K changes with a new input, Triton will check which config is best all over again
@triton.autotune(configs=autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_a_M, stride_a_K,
        stride_b_K, stride_b_N,
        stride_c_M, stride_c_N,
        # meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
):
    # we start with a 1D launch grid that we will turn into a 2D grid with a complicated "group-wise" ordering
    PID = tl.program_id(axis=0)
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N)

    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    group_id = PID // num_PID_in_group
    first_PID_along_M = group_id * GROUP_SIZE

    group_size_adj = min(num_PID_along_M - first_PID_along_M, GROUP_SIZE)

    PID_M = first_PID_along_M + ((PID % num_PID_in_group) % group_size_adj)
    PID_N = (PID % num_PID_in_group) // group_size_adj

    offset_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offset_K = tl.arange(0, BLOCK_SIZE_K)

    a_offset = offset_M[:, None] * stride_a_M + offset_K[None, :] * stride_a_K
    b_offset = offset_K[:, None] * stride_b_K + offset_N[None, :] * stride_b_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # the full C is shape (M, N)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = offset_K < K - k * BLOCK_SIZE_K
        # a_offset 是M * K，mask应该符合K mask[None,:]指在第0维插入新轴
        a = tl.load(a_ptr + a_offset, mask[None, :], other=0.0)
        b = tl.load(b_ptr + b_offset, mask[:, None], other=0.0)

        accumulator = tl.dot(a, b, acc=accumulator)

        a_offset += BLOCK_SIZE_K * stride_a_K
        b_offset += BLOCK_SIZE_K * stride_b_K

    # 写入目标位置
    c_offset = offset_M[:, None] * stride_c_M + offset_N[None, :] * stride_c_N
    c_mask = (offset_M[:, None] < M) & (offset_N[None, :] < N)
    tl.store(c_ptr + c_offset, accumulator, c_mask)


######### Step 2 #########
def matmul(a, b):
    # check constraints
    assert a.ndim == b.ndim == 2, "only supports matrices, not vectors or tensors"
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    # assert a.is_contiguous() and b.is_contiguous, "input matrices must be contiguous"
    a, b = a.to(torch.float16), b.to(torch.float16)

    # get dimesion lengths
    (M, K), (_, N) = a.shape, b.shape

    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # cdiv(x, y) = (x + (y - 1)) // y
    # A naive (slow) launch grid might try to separate our axes of parallelizatio into 2 dimensions, one
    #  for cdiv(M, BLOCK_SIZE_M) and the other for cdiv(N, BLOCK_SIZE_N)
    # Here instead we use a 1D launch kernel defined by cdiv(M, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N)
    # The reasoning behind this is explained inside the kernel
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


######### Step 1 #########
def matmul_kernel(size: tuple, atol=1e-2, rtol=1e-1, device=DEVICE):  # TODO does rtol=0 mean we don't use rtol?
    """
    Here is where we test the wrapper function and kernel that we wrote
    above to ensure all our values are correct, using pytorch as the
    correct answer to compare against

    We use higher tolerance values than previous tests because all the flop
    accumulation can really compound when it comes to a matmul; even slight
    differences in the block size and launch grid ordering from what PyTorch
    does can result in pretty sizeable discrepancies
    """
    # create input data
    torch.manual_seed(0)
    assert type(size) == tuple and len(size) == 2
    a = torch.randn((16, 8), device=DEVICE, dtype=torch.float16)
    b = torch.randn((8, 16), device=DEVICE, dtype=torch.float16)
    # run kernel & pytorch reference implementation
    c_tri = matmul(a, b)
    c_ref = torch.matmul(a, b)
    # compare
    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
    print("PASSED")


######### Step 4 #########
configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # we can increase multiple dimensions simultaneously while benchmarking
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch", "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={},
    )
]


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.05, 0.95]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 3 * M * N * K * 1e-12 / (ms * 1e-3)
    # 3 = number of memory operations (2 read + 1 write)
    # M * N * K = number of elements per memory op
    # 1e-12 converts flops to Teraflops
    # 1e-3 converts milliseconds to seconds
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    # always run unit-tests
    matmul_kernel(size=(1024, 1024))

    # Only run benchmark if explicitly requested
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)
