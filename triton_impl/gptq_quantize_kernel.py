import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
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
def _gate_up_silu_kernel(
        x_ptr,
        up_weight_ptr, up_zeros_ptr, up_scales_ptr, up_bias_ptr,
        gate_weight_ptr, gate_zeros_ptr, gate_scales_ptr, gate_bias_ptr,
        output_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wm, stride_wk,
        stride_om, stride_on,
        bits, is_signed,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = row < M
    col_mask = col < N

    accumulator_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        row_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = row_k < K

        x_block = tl.load(
            x_ptr + row[:, None] * stride_xm + row_k[None, :] * stride_xk,
            mask=row_mask[:, None] & mask_k[None, :],
            other=0.0
        )
        gate_w_fp16 = dequantize_block(
            gate_weight_ptr, gate_scales_ptr, gate_zeros_ptr,
            col, row_k, stride_wm, stride_wk, bits
        )
        up_w_fp16 = dequantize_block(
            up_weight_ptr, up_scales_ptr, up_zeros_ptr,
            col, row_k, stride_wm, stride_wk, bits
        )
        accumulator_gate += tl.dot(x_block, gate_w_fp16)
        accumulator_up += tl.dot(x_block, up_w_fp16)

    # --- 处理 Bias 部分 ---
    # 加载针对 N 维度的 bias [BN]
    # 如果 bias 为空（None），在外部调用时传 0 或控制逻辑
    b_gate = tl.load(gate_bias_ptr + col, mask=col_mask, other=0.0).to(tl.float32)
    b_up = tl.load(up_bias_ptr + col, mask=col_mask, other=0.0).to(tl.float32)

    # 将 [BN] 广播到 [BM, BN] 并累加
    gate_f32 = accumulator_gate + b_gate[None, :]
    up_f32 = accumulator_up + b_up[None, :]

    # 计算 SwiGLU: SiLU(gate) * up
    fused_f32 = (gate_f32 * tl.sigmoid(gate_f32)) * up_f32

    tl.store(
        output_ptr + row[:, None] * stride_om + col[None, :] * stride_on,
        fused_f32.to(tl.float16),
        mask=row_mask[:, None] & col_mask[None, :]
    )


@triton.jit
def dequantize_block(
        w_ptr,  # 指向打包后的 int32 权重矩阵 [K/8, N]
        s_ptr,  # 指向 Scales [N]
        z_ptr,  # 指向 Zeros [N] (注意：GPTQ 的 zeros 往往也是打包或偏移过的)
        ri,  # 当前 Block 的列索引 [BN]
        rk,  # 当前 Block 的行索引 [BK] (新增参数，用于定位 K 轴偏移)
        stride_wm,  # 权重的行步长
        stride_wk,  # 权重的列步长
        bits: tl.constexpr
):
    group_size = 32 // bits
    k_idx = rk // group_size
    shift = (rk % group_size) * bits
    packed_w = tl.load(w_ptr + k_idx[:, None] * stride_wm + ri[None, :] * stride_wk)

    mask = (1 << bits) - 1
    w_int4 = (packed_w >> shift[:, None]) & mask
    scales = tl.load(s_ptr + ri)  # [BN]
    zeros = tl.load(z_ptr + ri)  # [BN]
    w_fp16 = (w_int4.to(tl.float32) - zeros[None, :].to(tl.float32)) * scales[None, :].to(tl.float32)
    return w_fp16.to(tl.float16)


def fused_gate_up(hidden_state: torch.Tensor, gate, up) -> torch.Tensor:
    q_weight_gate = gate.qweight
    q_zeros_gate = gate.qzeros
    q_scales_gate = gate.qscales

    q_weight_up = up.qweight
    q_zeros_up = up.zeros
    q_scales_up = up.scales
    assert q_weight_gate.shape == q_weight_up.shape
    assert q_zeros_gate.shape == q_zeros_up.shape
    assert q_scales_gate == q_scales_up.shape
    assert q_zeros_gate.shape[0] * 128 == q_weight_gate.shape[0] * (32 // q_weight_gate.bits)

    original_shape = hidden_state.shape
    if hidden_state.dim() == 3:
        x_2d = hidden_state.view(-1, original_shape[-1])
    else:
        x_2d = hidden_state

    M, K = x_2d.shape
    # moe_intermediate_size
    N = gate.qweight.shape[1]
    output = torch.empty((M, N), dtype=torch.float16)
    gate_bias = gate.bias if hasattr(gate, 'bias') and gate.bias is not None else torch.zeros(N, device=DEVICE,
                                                                                              dtype=torch.float16)
    up_bias = up.bias if hasattr(up, 'bias') and up.bias is not None else torch.zeros(N, device=DEVICE,
                                                                                      dtype=torch.float16)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    _gate_up_silu_kernel[grid](
        x_ptr=x_2d,
        up_weight_ptr=up.qweight,
        up_zeros_ptr=up.qzeros,
        up_scales_ptr=up.qscales,
        up_bias_ptr=up.bias,
        gate_weight_ptr=gate.qweight,
        gate_zeros_ptr=gate.qzeros,
        gate_scales_ptr=gate.qscales,
        gate_bias_ptr=gate.bias,
        output_ptr=output,
        M=M, N=N, K=K,
        stride_xm=x_2d.stride(0), stride_xk=x_2d.stride(1),
        stride_wm=gate.qweight.stride(0), stride_wk=gate.qweight.stride(1),
        stride_om=output.stride(0), stride_on=output.stride(1),
        bits=gate.bits,
        is_signed=False
    )
    if hidden_state.dim() == 3:
        return output.view(original_shape[0], original_shape[1], N)
    return output
