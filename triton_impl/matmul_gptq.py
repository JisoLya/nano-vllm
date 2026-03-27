import torch
import triton
import triton.language as tl
from numpy import dtype

from nanovllm.layers.quantized_linear import GPTQLinear


@triton.jit
def _matmul_gptq_kernel(
        x_ptr,
        M, K, N,
        output_ptr,
        y_weight_ptr,
        y_zeros_ptr,
        y_scales_ptr,
        y_bias_ptr,
        stride_x_m, stride_x_n,
        stride_o_k,
        stride_w_n, stride_w_k,
        stride_s_g, stride_s_n,  # scales的步长
        stride_z_g, stride_z_k,  # zeros的步长
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    row_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    mask_m = row_m < M
    mask_k = col_k < K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for n in range(0, N, BLOCK_SIZE_N):
        # 遍历hidden_states
        col_n = n + tl.arange(0, BLOCK_SIZE_N)
        mask_n = col_n < N
        x_block = tl.load(x_ptr + row_m[:, None] * stride_x_m + col_n[None, :] * stride_x_n,
                          mask=mask_m[:, None] & mask_n[None, :],
                          other=0.0).to(tl.float32)

        # 加载这一块对应的weight
        w_fp32 = dequantize_block(
            y_weight_ptr, y_scales_ptr, y_zeros_ptr,
            col_n, col_k,
            stride_w_k, stride_w_n,
            stride_s_g, stride_s_n,
            stride_z_g, stride_z_k,
            bits=4
        )

        accumulator += tl.dot(x_block, w_fp32)

    if y_bias_ptr is not None:
        bias = tl.load(y_bias_ptr + col_k, mask=mask_k, other=0.0).to(tl.float32)
        accumulator += bias[None, :]

    tl.store(output_ptr + row_m[:, None] * stride_x_m + col_k[None, :] * stride_o_k,
             accumulator.to(tl.float16),
             mask=mask_m[:, None] & mask_k[None, :]
             )


@triton.jit
def dequantize_block(
        weight_ptr, scales_ptr, zeros_ptr,
        rn, rk,  # rn -> N维, rk -> K维
        stride_w_k, stride_w_n,  # 注意顺序：现在是[K/8, N]
        stride_s_g, stride_s_k,  # scales: [N/128, N]
        stride_z_g, stride_z_k,  # zeros: [N/128, N/8]
        bits: tl.constexpr
):
    group_size = 32 // bits  # 8

    k_idx = rk // group_size
    shift = (rk % group_size) * bits

    pack_w = tl.load(
        weight_ptr + rn[:, None] * stride_w_n + k_idx[None, :] * stride_w_k
    )
    mask = (1 << bits) - 1
    w_int = (pack_w >> shift[None, :]) & 0xF

    g_idx = rn // 128
    scales = tl.load(scales_ptr + g_idx[:, None] * stride_s_g + rk[None, :] * stride_s_k)

    z_k_idx = rk // 8
    z_shift = (rk % 8) * 4
    pack_z = tl.load(zeros_ptr + g_idx[:, None] * stride_z_g + z_k_idx[None, :] * stride_z_k)
    z_int = (pack_z >> z_shift[None, :]) & 0xF
    return (w_int.to(tl.float32) - z_int.to(tl.float32)) * scales.to(tl.float32)


# x[M, intermediate_size]
def matmul_gptq(x: torch.Tensor, down_proj: GPTQLinear) -> torch.Tensor:
    M, N = x.shape
    N, K = down_proj.shape

    output = torch.empty((M, K), dtype=torch.float16, device=x.device)

    grid = lambda meta: (
        (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(K, meta['BLOCK_SIZE_K'])),
    )

    _matmul_gptq_kernel[grid](

    )

    return output
