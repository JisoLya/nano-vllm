import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_gptq_kernel(
        x_ptr, output_ptr, y_weight_ptr, y_zeros_ptr, y_scales_ptr, y_bias_ptr,
        M, N, K,  # 统一顺序：M是行，N是缩减维度，K是输出维度
        stride_xm, stride_xn,
        stride_ok, stride_om,
        stride_wn, stride_wk,  # 这里的 stride_wn 指向打包后的 N/8 维度
        stride_sg, stride_sk,
        stride_zg, stride_zk,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(0, N, BLOCK_SIZE_N):
        # 遍历hidden_states
        rn = n + tl.arange(0, BLOCK_SIZE_N)
        x_block = tl.load(x_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn,
                          mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)

        w_fp32 = dequantize_block(
            y_weight_ptr, y_scales_ptr, y_zeros_ptr,
            rn, rk,
            stride_wn, stride_wk,
            stride_sg, stride_sk,
            stride_zg, stride_zk,
        )

        accumulator += tl.dot(x_block.to(tl.float32), w_fp32, out_dtype=tl.float32)

    if y_bias_ptr is not None:
        bias = tl.load(y_bias_ptr + rk, mask=(rk < K), other=0.0)
        accumulator += bias[None, :]

    tl.store(output_ptr + rm[:, None] * stride_om + rk[None, :],
             accumulator.to(tl.float16),
             mask=(rm[:, None] < M) & (rk[None, :] < K))


@triton.jit
def dequantize_block(
        weight_ptr, scales_ptr, zeros_ptr,
        rn, rk,
        stride_wn, stride_wk,
        stride_sg, stride_sk,
        stride_zg, stride_zk,
):
    n_idx = rn // 8
    n_shift = (rn % 8) * 4

    offs_w = n_idx[:, None] * stride_wn + rk[None, :] * stride_wk
    pack_w = tl.load(weight_ptr + offs_w)
    w_int = (pack_w >> n_shift[:, None]) & 0xF

    g_idx = rn // 128
    k_idx = rk // 8
    k_shift = (rk % 8) * 4

    offs_z = g_idx[:, None] * stride_zg + k_idx[None, :] * stride_zk
    pack_z = tl.load(zeros_ptr + offs_z)
    z_int = (pack_z >> k_shift[None, :]) & 0xF

    offs_s = g_idx[:, None] * stride_sg + rk[None, :] * stride_sk
    scales = tl.load(scales_ptr + offs_s)

    return (w_int.to(tl.float32) - z_int.to(tl.float32)) * scales.to(tl.float32)


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
        down_proj.qweight.stride(0), down_proj.qweight.stride(1),
        down_proj.scales.stride(0), down_proj.scales.stride(1),
        down_proj.qzeros.stride(0), down_proj.qzeros.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64  # 常见的性能参数
    )
    return output
