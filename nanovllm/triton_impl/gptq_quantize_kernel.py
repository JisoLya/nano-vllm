import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16}, num_stages=1, num_warps=2),
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
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = 8 * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * 8
    group_size_m = min(num_pid_m - first_pid_m, 8)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 2. 偏移计算
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N

    accumulator_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        ).to(tl.float16)

        # 解包 Gate 权重
        w_gate = dequantize_block(
            gate_weight_ptr, gate_scales_ptr, gate_zeros_ptr,
            k, offs_n, stride_wk, stride_wn, stride_zk, stride_zn, stride_sk, stride_sn,
            K, N, BLOCK_SIZE_K, BLOCK_SIZE_N
        )

        # 解包 Up 权重
        w_up = dequantize_block(
            up_weight_ptr, up_scales_ptr, up_zeros_ptr,
            k, offs_n, stride_wk, stride_wn, stride_zk, stride_zn, stride_sk, stride_sn,
            K, N, BLOCK_SIZE_K, BLOCK_SIZE_N
        )

        # Tensor Core 矩阵乘法
        accumulator_gate += tl.dot(x, w_gate)
        accumulator_up += tl.dot(x, w_up)

    b_gate = tl.load(gate_bias_ptr + col, mask=col_mask, other=0.0).to(tl.float32)
    b_up = tl.load(up_bias_ptr + col, mask=col_mask, other=0.0).to(tl.float32)

    # 将 [BN] 广播到 [BM, BN] 并累加
    gate_f32 = accumulator_gate + b_gate[None, :]
    up_f32 = accumulator_up + b_up[None, :]

    # 计算 SwiGLU: SiLU(gate) * up
    fused_f32 = (gate_f32 * tl.sigmoid(gate_f32)) * up_f32

    tl.store(
        output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        fused_f32.to(tl.float16),
        mask=mask_m[:, None] & mask_n[None, :]
    )


@triton.jit
def dequantize_block(
        w_ptr, scales_ptr, zeros_ptr,
        k_idx, col_idx,
        stride_wk, stride_wn,
        stride_zk, stride_zn,
        stride_sk, stride_sn,
        K, N,
        BLOCK_SIZE_K, BLOCK_SIZE_N,
):
    rk_packed = (k_idx // 8) + tl.arange(0, BLOCK_SIZE_K // 8)
    w_offs = rk_packed[:, None] * stride_wk + col_idx[None, :] * stride_wn
    w_mask = (rk_packed[:, None] < (K//8)) & (col_idx[None, :] < N)
    
    w_packed = tl.load(w_ptr + w_offs, mask=w_mask,other=0.0)
    
    shifts = (tl.arange(0, 8) * 4)[None, :,None]
    w_unpacked = (w_packed[:, None, :] >> shifts) & 0xF
    w_unpacked = tl.reshape(w_unpacked, [BLOCK_SIZE_K, BLOCK_SIZE_N])
    
    group_idx = k_idx // 8
    col_packed = (col_idx // 8)
    z_offsets = group_idx * stride_zk + col_packed[None, :] * stride_zn
    z_mask = col_packed[None, :] < (N // 8)
    z_packed = tl.load(zeros_ptr + z_offsets, mask=z_mask, other=0)

    shifts_n = (tl.arange(0, 8) * 4)[None, None, :]
    z_unpacked = (z_packed[:, :, None] >> shifts_n) & 0xF
    z_unpacked = tl.reshape(z_unpacked, (1, BLOCK_SIZE_N))

    s_offsets = group_idx * stride_sk + col_idx[None, :] * stride_sn
    s_mask = col_idx[None, :] < N
    s = tl.load(scales_ptr + s_offsets, mask=s_mask, other=1.0) # (1, BN)

    w_fp16 = (w_unpacked.to(tl.float16) - (z_unpacked.to(tl.float16) + 1.0)) * s.to(tl.float16)
    return w_fp16


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
