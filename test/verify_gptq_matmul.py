import torch

from test.mock.mock_GPTQLinear import MockGPTQLinear
from triton_impl.matmul_gptq import matmul_gptq
from verify_fused_gate_up import prepare_data, prepare_hidden_size, unpack_wzs_torch


def verify_gptq_matmul():
    torch.manual_seed(0)
    w, z, s = prepare_data()
    hidden_size = prepare_hidden_size(2560, 3584)

    w_torch = unpack_wzs_torch(w, z, s).to(torch.float16)

    ref = (hidden_size @ w_torch).to(torch.float16)

    mock_w = MockGPTQLinear(w, z, s)

    tri = matmul_gptq(hidden_size, mock_w)

    torch.testing.assert_close(ref, tri, atol=1e-2, rtol=1e-1)

    print("Gptq Matmul PASSED")


if __name__ == '__main__':
    verify_gptq_matmul()
