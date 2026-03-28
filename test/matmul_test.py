import torch

from triton_impl.matmul_gptq import matmul_gptq


# 假设你的 GPTQLinear 类定义如下
class MockGPTQLinear:
    def __init__(self, qweight, qzeros, scales, bias):
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.bias = bias


def matmul_gptq_correctness(atol=2e-1, rtol=1e-1):
    device = torch.device("cuda")
    torch.manual_seed(42)
    # qweight: [448, 2560] -> 448 * 8 = 3584 (N), 2560 (K)
    # qzeros: [28, 320] -> 28 groups, 320 * 8 = 2560 (K)
    M, N, K = 16, 3584, 2560
    group_size = 128  # 3584 // 28 = 128

    print(f"Testing Matmul GPTQ: M={M}, N={N}, K={K}, GroupSize={group_size}")

    raw_w = torch.randint(0, 16, (N, K), dtype=torch.int32, device=device)

    qweight = torch.zeros((N // 8, K), dtype=torch.int32, device=device)
    for i in range(448):
        for j in range(8):
            qweight[i, :] |= (raw_w[i * 8 + j, :] << (j * 4))

    raw_z = torch.randint(0, 16, (N // group_size, K), dtype=torch.int32, device=device)

    qzeros = torch.zeros((N // group_size, K // 8), dtype=torch.int32, device=device)
    reshaped_z = raw_z.view(N // group_size, K // 8, 8)
    for i in range(8):
        qzeros |= (reshaped_z[:, :, i].to(torch.int32) << (i * 4))

    scales = torch.randn((N // group_size, K), dtype=torch.float16, device=device)
    bias = torch.randn((K,), dtype=torch.float16, device=device)

    down_proj = MockGPTQLinear(qweight, qzeros, scales, bias)

    x = torch.randn((M, N), dtype=torch.float16, device=device)

    full_scales = scales.repeat_interleave(group_size, dim=0)
    full_zeros = raw_z.repeat_interleave(group_size, dim=0)

    ref_w_fp = (raw_w.float() - full_zeros.float()) * full_scales.float()

    c_ref = torch.matmul(x.float(), ref_w_fp) + bias.float()
    c_ref = c_ref.to(torch.float16)

    try:
        c_tri = matmul_gptq(x, down_proj)

        torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
        print("✅ PASSED: Triton output matches PyTorch reference!")
    except Exception as e:
        print(f"❌ FAILED: {e}")


'''
Testing Matmul GPTQ: M=16, N=3584, K=2560, GroupSize=128
❌ FAILED: Tensor-likes are not close!
Mismatched elements: 13 / 40960 (0.0%)
Greatest absolute difference: 0.369140625 at index (10, 1214) (up to 0.1 allowed)
Greatest relative difference: 2.6640625 at index (1, 1666) (up to 0.1 allowed)
'''
# 一直有0.1的累计误差
if __name__ == "__main__":
    matmul_gptq_correctness()
