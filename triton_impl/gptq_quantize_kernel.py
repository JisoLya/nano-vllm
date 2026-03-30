import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=2),
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
        stride_wk, stride_wn,
        stride_om, stride_on,
        stride_zk, stride_zn,
        stride_sk, stride_sn,
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
        rk = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = rk < K

        x = tl.load(x_ptr + row[:, None] * stride_xm + rk[None, :] * stride_xk,
                    mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        w_gate = dequantize_block(
            gate_weight_ptr, gate_scales_ptr, gate_zeros_ptr,
            rk, col, stride_wk, stride_wn, stride_zk, stride_zn, stride_sk, stride_sn,
            k_mask, col_mask
        )

        # 实时解包 Up 权重块: [BK, BN]
        w_up = dequantize_block(
            up_weight_ptr, up_scales_ptr, up_zeros_ptr,
            rk, col, stride_wk, stride_wn, stride_zk, stride_zn, stride_sk, stride_sn,
            k_mask, col_mask
        )
        accumulator_gate += tl.dot(x.to(tl.float16), w_gate.to(tl.float16))
        accumulator_up += tl.dot(x.to(tl.float16), w_up.to(tl.float16))

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
        ptr, scales_ptr, zeros_ptr,
        row_k, col,
        stride_wk, stride_wn,
        stride_zk, stride_zn,
        stride_sk, stride_sn,
        row_mask, col_mask
):
    wk_packed = row_k // 8
    wk_shift = (row_k % 8) * 4

    zn_packed = col // 8
    zn_shift = (col % 8) * 4

    group_idx = row_k // 128

    w_offsets = wk_packed[:, None] * stride_wk + col[None, :] * stride_wn
    w_packed = tl.load(ptr + w_offsets, mask=row_mask[:, None] & col_mask[None, :], other=0)

    # z_offsets: (BLOCK_SIZE_K, BLOCK_SIZE_N) -> group_idx, zn_packed
    z_offsets = group_idx[:, None] * stride_zk + zn_packed[None, :] * stride_zn
    z_packed = tl.load(zeros_ptr + z_offsets, mask=row_mask[:, None] & col_mask[None, :], other=0)

    # s_offsets: (BLOCK_SIZE_K, BLOCK_SIZE_N)
    s_offsets = group_idx[:, None] * stride_sk + col[None, :] * stride_sn
    s = tl.load(scales_ptr + s_offsets, mask=row_mask[:, None] & col_mask[None, :], other=1.0)

    # 3. 解包与计算
    w_unpacked = (w_packed >> wk_shift[:, None]) & 0xF
    z_unpacked = (z_packed >> zn_shift[None, :]) & 0xF

    # 返回 FP32 结果供 tl.dot 使用
    return (w_unpacked.to(tl.float32) - z_unpacked.to(tl.float32)) * s.to(tl.float32)


def fused_gate_up(hidden_state: torch.Tensor, gate, up) -> torch.Tensor:
    q_weight_gate = gate.qweight
    q_zeros_gate = gate.qzeros
    q_scales_gate = gate.scales

    q_weight_up = up.qweight
    q_zeros_up = up.qzeros
    q_scales_up = up.scales
    assert q_weight_gate.shape == q_weight_up.shape
    assert q_zeros_gate.shape == q_zeros_up.shape
    assert q_scales_gate.shape == q_scales_up.shape
    assert q_zeros_gate.shape[0] * 128 == q_weight_gate.shape[0] * 8

    original_shape = hidden_state.shape
    if hidden_state.dim() == 3:
        x_2d = hidden_state.view(-1, original_shape[-1])
    else:
        x_2d = hidden_state

    M, K = x_2d.shape
    # moe_intermediate_size
    N = gate.qweight.shape[1]
    output = torch.empty((M, N), dtype=torch.float16, device=DEVICE)
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
        up_scales_ptr=up.scales,
        up_bias_ptr=up_bias,
        gate_weight_ptr=gate.qweight,
        gate_zeros_ptr=gate.qzeros,
        gate_scales_ptr=gate.scales,
        gate_bias_ptr=gate_bias,
        output_ptr=output,
        M=M, N=N, K=K,
        stride_xm=x_2d.stride(0), stride_xk=x_2d.stride(1),
        stride_wk=gate.qweight.stride(0), stride_wn=gate.qweight.stride(1),
        stride_om=output.stride(0), stride_on=output.stride(1),
        stride_zk=gate.qzeros.stride(0), stride_zn=gate.qzeros.stride(1),
        stride_sk=gate.scales.stride(0), stride_sn=gate.scales.stride(1),
    )
    if hidden_state.dim() == 3:
        return output.view(original_shape[0], original_shape[1], N)
    return output
