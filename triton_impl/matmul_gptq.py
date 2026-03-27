import torch
import triton

from nanovllm.layers.quantized_linear import GPTQLinear


@triton.jit
def _matmul_gptq_kernel(
        x_ptr,
        M, N,
        y_weight_ptr,
        y_zeros_ptr,
        y_scales_ptr,
        y_bias_ptr,
):
    pass


# x[M, intermediate_size]
def matmul_gptq(x: torch.Tensor, down_proj: GPTQLinear) -> torch.Tensor:
    M, N = x.shape
    N, K = down_proj.shape

    output = torch.empty((M, K), dtype=torch.float16, device=x.device)

    grid = lambda meta: (

    )
    _matmul_gptq_kernel[grid](

    )

    return output
