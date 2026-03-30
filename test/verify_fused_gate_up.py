import torch
import triton
import triton.language as tl

from nanovllm.layers.activation import SiluAndMul
from test.mock.mock_GPTQLinear import MockGPTQLinear
from triton_impl.gptq_quantize_kernel import fused_gate_up


def prepare_data(device="cuda"):
    weight = torch.randint(
        0, 0xFFFFFFFF + 1,
        size=(448, 2560),
        dtype=torch.int64,
        device=device
    ).to(torch.int32)

    zeros = torch.randint(
        0, 0xFFFFFFFF + 1,
        size=(28, 320),
        dtype=torch.int64,
        device=device
    ).to(torch.int32)

    scales = torch.empty((28, 2560), dtype=torch.float16, device=device)
    scales.uniform_(0.005, 0.02)

    return weight, zeros, scales


def unpack_wzs_torch(w: torch.Tensor,
                     z: torch.Tensor,
                     s: torch.Tensor) -> torch.Tensor:
    original_weight = torch.empty((3584, 2560), device="cuda")
    for i in range(w.shape[0]):
        for j in range(8):
            original_weight[i * 8 + j, :] = (w[i] >> (j * 4)) & 0xf

    # 把z按列依次4bit拆开
    unpack_z = torch.empty((28, 2560), device="cuda")
    for i in range(z.shape[0]):
        for j in range(320):
            for k in range(8):
                unpack_z[i, j * 8 + k] = (z[i, j] >> (k * 4)) & 0xf

    for i in range(unpack_z.shape[0]):
        original_weight[i * 128: (i + 1) * 128, :] -= unpack_z[i]

    for i in range(s.shape[0]):
        original_weight[i * 128: (i + 1) * 128, :] *= s[i, :]

    return original_weight


@triton.jit
def de_quantize_block_kernel(
        w_ptr, z_ptr, s_ptr, o_ptr,
        M, N,  # 目标矩阵的维度 (3584, 2560)
        stride_wk, stride_wn,
        stride_zk, stride_zn,  # 增加 z 的列步长
        stride_sk, stride_sn,  # 增加 s 的列步长
        stride_ok, stride_on,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    m_mask = rm < M
    n_mask = rn < N

    rk_packed = rm // 8
    rk_shift = (rm % 8) * 4

    group_idx = rm // 128
    rn_packed = rn // 8
    rn_shift = (rn % 8) * 4

    # 加载权重
    w_offsets = rk_packed[:, None] * stride_wk + rn[None, :] * stride_wn
    w_packed = tl.load(w_ptr + w_offsets, mask=m_mask[:, None] & n_mask[None, :], other=0)

    z_offsets = group_idx[:, None] * stride_zk + rn_packed[None, :] * stride_zn
    z_packed = tl.load(z_ptr + z_offsets, mask=m_mask[:, None] & n_mask[None, :], other=0)
    s_offsets = group_idx[:, None] * stride_sk + rn[None, :] * stride_sn
    s_packed = tl.load(s_ptr + s_offsets, mask=m_mask[:, None] & n_mask[None, :], other=1.0)

    w_unpacked = (w_packed >> rk_shift[:, None]) & 0xf
    z_unpacked = (z_packed >> rn_shift[None, :]) & 0xf

    out = (w_unpacked.to(tl.float32) - z_unpacked.to(tl.float32)) * s_packed.to(tl.float32)

    out_offsets = rm[:, None] * stride_ok + rn[None, :] * stride_on
    tl.store(o_ptr + out_offsets, out.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])


def unpack_w_triton(w: torch.Tensor,
                    z: torch.Tensor,
                    s: torch.Tensor) -> torch.Tensor:
    output = torch.empty((3584, 2560), device="cuda")
    M, N = output.shape

    grid = lambda meta: (
        (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N'])
         )
    )
    # [M, K] @ [K, N] -> [M, N]
    # x @ weight -> output

    # 这里K = w.shape[0] * 8, (M, N) -> (3584, 2560)
    de_quantize_block_kernel[grid](
        w, z, s, output,
        M, N,
        w.stride(0), w.stride(1),
        z.stride(0), z.stride(1),
        s.stride(0), s.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32
    )
    return output


def verify_dequantize(atol=1e-2, rtol=1e-1):
    torch.manual_seed(0)
    w, z, s = prepare_data()

    w_torch = unpack_wzs_torch(w, z, s)
    w_triton = unpack_w_triton(w, z, s)

    torch.testing.assert_close(w_torch, w_triton, atol=atol, rtol=rtol)
    print("Verify De-quantize PASSED")


def prepare_hidden_size(M, N, device="cuda"):
    return torch.randn((M, N), dtype=torch.float16, device=device).clamp(-4.0, 4.0)


def verify_gate_up(atol=1e-2, rtol=1e-1):
    torch.manual_seed(0)
    hidden_size = prepare_hidden_size(2560, 3584) * 0.001

    g_w, g_z, g_s = prepare_data()
    u_w, u_z, u_s = prepare_data()

    gate = unpack_wzs_torch(g_w, g_z, g_s)
    up = unpack_wzs_torch(u_w, u_z, u_s)

    gate_up = torch.cat(
        (hidden_size @ gate, hidden_size @ up),
        dim=-1).to(torch.float16)
    fused_kernel = SiluAndMul()
    ref = fused_kernel(gate_up)

    quantize_gate = MockGPTQLinear(g_w, g_z, g_s)
    quantize_up = MockGPTQLinear(u_w, u_z, u_s)

    tri = fused_gate_up(hidden_size, quantize_gate, quantize_up)

    torch.testing.assert_close(ref, tri, atol=atol, rtol=rtol)
    print("Verify Gate-Up PASSED")


if __name__ == "__main__":
    # verify_dequantize()
    verify_gate_up()
