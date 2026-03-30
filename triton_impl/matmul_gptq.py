import torch
import triton
import triton.language as tl

from triton_impl.gptq_quantize_kernel import dequantize_block

autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3,
                  num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4,
                  num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4,
                  num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4,
                  num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4,
                  num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4,
                  num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5,
                  num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5,
                  num_warps=2)
]


@triton.autotune(autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _matmul_gptq_kernel(
        x_ptr, output_ptr, y_weight_ptr, y_zeros_ptr, y_scales_ptr, y_bias_ptr,
        M, N, K,
        stride_xm, stride_xn,
        stride_ok, stride_om,
        stride_wk, stride_wn,  # 接收 qweight 的 strides
        stride_sk, stride_sn,  # 接收 scales 的 strides
        stride_zk, stride_zn,  # 接收 qzeros 的 strides
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(0, N, BLOCK_SIZE_N):
        rn = n + tl.arange(0, BLOCK_SIZE_N)

        # 1. 生成 mask (解决越界产生 inf 的核心)
        x_row_mask = rm[:, None] < M
        w_row_mask = rn < N
        col_mask = rk < K

        x_block = tl.load(x_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn,
                          mask=x_row_mask & w_row_mask[None, :], other=0.0)

        w_fp32 = dequantize_block(
            y_weight_ptr, y_scales_ptr, y_zeros_ptr,
            rn, rk,
            stride_wk, stride_wn,  # 对应 dequantize_block 里的 weight stride
            stride_zk, stride_zn,  # 对应 dequantize_block 里的 zero stride
            stride_sk, stride_sn,  # 对应 dequantize_block 里的 scale stride
            w_row_mask, col_mask  # 补充上原先漏传的 mask！避免读取越界垃圾数据
        )

        accumulator += tl.dot(x_block.to(tl.float32), w_fp32, out_dtype=tl.float32)

    if y_bias_ptr is not None:
        bias = tl.load(y_bias_ptr + rk, mask=(rk < K), other=0.0)
        accumulator += bias[None, :]

    tl.store(output_ptr + rm[:, None] * stride_om + rk[None, :],
             accumulator.to(tl.float16),
             mask=(rm[:, None] < M) & (rk[None, :] < K))


def matmul_gptq(x: torch.Tensor, down_proj) -> torch.Tensor:
    M, N = x.shape
    _, K = down_proj.qweight.shape

    output = torch.empty((M, K), dtype=torch.float16, device=x.device)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(K, meta['BLOCK_SIZE_K'])
    )

    _matmul_gptq_kernel[grid](
        x, output,
        down_proj.qweight, down_proj.qzeros, down_proj.scales, down_proj.bias,
        M, N, K,
        x.stride(0), x.stride(1),
        output.stride(1), output.stride(0),

        # 按正确的顺序传给 Kernel
        down_proj.qweight.stride(0), down_proj.qweight.stride(1),
        down_proj.scales.stride(0), down_proj.scales.stride(1),
        down_proj.qzeros.stride(0), down_proj.qzeros.stride(1),
    )
    return output
