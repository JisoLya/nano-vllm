import torch

import triton
import triton.language as tl

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
def dequantize_block(
    w_ptr, z_ptr, s_ptr,
    output_ptr, M, N,
    stride_w0, stride_w1,
    stride_z0, stride_z1,
    stride_s0, stride_s1,
    stride_o0,stride_o1,
    BLOCK_SIZE_M :tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr,
    GROUP_SIZE
    ):
    pid = tl.program_id(0)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_block_n
    pid_n = pid % num_block_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)    
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    w_offs = (offs_m // 8)[:, None] * stride_w0 + offs_n[None, :] * stride_w1
    w_mask = ((offs_m // 8) < (M // 8))[:, None] & offs_n[None, :] < N
    w_packed = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)
    
    offs_zm = offs_m // 128
    offs_zn = offs_n // 8
    
    offs_z = offs_zm[:, None] * stride_z0 + offs_zn[None,:] * stride_z1
    mask_z = (offs_zm < (M // GROUP_SIZE))[:, None] & (offs_zn < (N // 8))[None,:]
    z_packed = tl.load(z_ptr + offs_z, mask=mask_z, other=0.0)
    
    offs_s = offs_zm[:,None] * stride_s0 + offs_n[None, :] * stride_s1
    mask_s = (offs_zm < (M // 128))[:, None] & mask_n[None, :]
    s_packed = tl.load(s_ptr + offs_s, mask=mask_s, other=0)
    
    # [0, 1, 2, 3,..., 15] -> [0, 0, 0, 0, ..., 4, 4, 4, 4]
    shift_w = (offs_m % 8) * 4
    w_unpack = (w_packed >> shift_w[:, None]) & 0xf
    shift_z = (offs_n % 8) * 4
    z_unpack = (z_packed >> shift_z[None, :]) & 0xf
    
    out = (w_unpack.to(tl.float32) - z_unpack.to(tl.float32)) * s_packed.to(tl.float32)
    out_offs = offs_m[:,None]*stride_o0 + offs_n[None, :]*stride_o1
    out_mask = mask_m[:, None] & mask_n[None,:]
    
    tl.store(output_ptr + out_offs, out, mask=out_mask)
    

def unpack_triton(w:torch.Tensor, z:torch.Tensor, s:torch.Tensor):
    # int4量化
    _, N = w.shape
    
    M = w.shape[0] * 8
    output = torch.empty([M, N], dtype=torch.float32, device=w.device)
    
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    
    
    dequantize_block[grid](
        w, z, s, output,
        M, N,
        w.stride(0),w.stride(1),
        z.stride(0),z.stride(1),
        s.stride(0),s.stride(1),
        output.stride(0),output.stride(1),
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        GROUP_SIZE=128)
    
    return output


if __name__ == "__main__":
    w, z, s = prepare_data("cuda")

    out_torch = unpack_wzs_torch(w, z, s)
    out_triton = unpack_triton(w, z, s)

    # 校验结果是否绝对对齐
    max_diff = torch.max(torch.abs(out_torch - out_triton)).item()
    print(f"Max difference between Torch and Triton: {max_diff}")
    
    assert torch.allclose(out_torch, out_triton, atol=1e-2, rtol=1e-2), "Results do not match!"
    print("✅ Dequantization Triton kernel verified successfully!")