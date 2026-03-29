import torch
import triton
import triton.language as tl


def prepare_data(device="cuda"):
    torch.manual_seed(0)
    weight = torch.randint(
        0, 0xFFFFFFF,
        size=(448, 2560),
        dtype=torch.int32,
        device=device
    )
    zeros = torch.randint(
        0, 0xFFFFFFF,
        size=(28, 320),
        dtype=torch.int32,
        device=device
    )
    scales = torch.randn(
        (28, 2560),
        dtype=torch.float16,
        device=device
    )
    return weight, zeros, scales


def unpack_wzs_torch(w: torch.Tensor,
                     z: torch.Tensor,
                     s: torch.Tensor) -> torch.Tensor:
    original_weight = torch.empty((3584, 2560))
    for i in range(w.shape[0]):
        for j in range(8):
            original_weight[i * 8 + j, :] = (w[i] >> (j * 4)) & 0xf

    # 把z按列依次4bit拆开
    unpack_z = torch.empty((28, 2560))
    for i in range(z.shape[0]):
        for j in range(320):
            for k in range(8):
                unpack_z[i, j * 8 + k] = (z[i, j] >> (k * 4)) & 0xf

    for i in range(unpack_z.shape[0]):
        original_weight[i * 128: (i + 1) * 128, :] -= unpack_z[i]

    for i in range(s.shape[0]):
        original_weight[i * 128: (i + 1) * 128, :] *= s[i, :]

    return original_weight


def unpack_w_triton(w: torch.Tensor,
                    z: torch.Tensor,
                    s: torch.Tensor) -> torch.Tensor:
    output = torch.empty((3584, 2560))
    K, N = output.shape

    gird = lambda meta: (
        (tl.cdiv(K, meta['BLOCK_SIZE_K']), tl.cdiv(N, meta['BLOCK_SIZE_N'])
         )
    )

    return output


if __name__ == "__main__":
    w, z, s = prepare_data()
    original = unpack_wzs_torch(w, z, s)
