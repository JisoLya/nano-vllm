import torch
import triton

from triton_impl.gptq_quantize_kernel import _gate_up_silu_kernel


def pack_int4(fp16_weights):
    K, N = fp16_weights.shape
    q_weights = torch.clamp(fp16_weights.to(torch.float32), 0, 15).to(torch.int32)

    packed = torch.zeros((K // 8, N), dtype=torch.int32, device=fp16_weights.device)
    for i in range(8):
        packed |= (q_weights[i::8, :] << (i * 4))
    return packed


def torch_reference(x, g_w, g_s, g_z, u_w, u_s, u_z):
    x = x.to(torch.float16)
    real_g_w = ((g_w - g_z[None, :]) * g_s[None, :]).to(torch.float16)
    real_u_w = ((u_w - u_z[None, :]) * u_s[None, :]).to(torch.float16)

    gate_act = torch.matmul(x, real_g_w)
    up_act = torch.matmul(x, real_u_w)

    res = torch.nn.functional.silu(gate_act) * up_act
    return res.to(torch.float16)


def verify():
    # 1. 参数设置
    M, K, N = 1, 3584, 18944  # Qwen2 标准维度
    device = "cuda"
    x = torch.randn((M, K), device=device, dtype=torch.float16)

    g_s = torch.rand((N,), device=device, dtype=torch.float16) * 0.1
    g_z = torch.randint(0, 15, (N,), device=device, dtype=torch.float16)
    u_s = torch.rand((N,), device=device, dtype=torch.float16) * 0.1
    u_z = torch.randint(0, 15, (N,), device=device, dtype=torch.float16)

    # 模拟原始权重并打包
    raw_g_w = torch.randint(0, 15, (K, N), device=device, dtype=torch.float16)
    raw_u_w = torch.randint(0, 15, (K, N), device=device, dtype=torch.float16)

    packed_g_w = pack_int4(raw_g_w)  # [448, 18944]
    packed_u_w = pack_int4(raw_u_w)  # [448, 18944]

    output = torch.zeros((M, N), device=device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N'])
    )
    u_bias = torch.zeros((N,), device=device, dtype=torch.float16)
    g_bias = torch.zeros((N,), device=device, dtype=torch.float16)

    _gate_up_silu_kernel[grid](
        x,
        packed_u_w, u_z, u_s, u_bias,  # 补上 u_bias
        packed_g_w, g_z, g_s, g_bias,  # 补上 g_bias
        output,
        M, N, K,
        x.stride(0), x.stride(1),
        packed_g_w.stride(0), packed_g_w.stride(1),
        output.stride(0), output.stride(1),
        bits=4,
        is_signed=False
    )

    ref_output = torch_reference(x, raw_g_w, g_s, g_z, raw_u_w, u_s, u_z)
    try:
        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
        print("✅ PASSED: Triton Kernel matches PyTorch reference!")
    except Exception as e:
        print(f"❌ FAILED: Accuracy mismatch\n{e}")


if __name__ == "__main__":
    verify()
