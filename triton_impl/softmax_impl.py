import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


@triton.jit
def _softmax_kernel(
        x_ptr, y_ptr,
        stride_x, stride_y,
        n_rows, n_cols,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)

    for idx in tl.range(pid, n_rows, step=step):
        row_start_ptr = x_ptr + idx * stride_x

        col_offsets = tl.arange(0, BLOCK_SIZE)
        inputs_ptr = row_start_ptr + col_offsets
        col_mask = col_offsets < n_cols
        row = tl.load(inputs_ptr, mask=col_mask, other=float('-inf'))
        row_minus_max = row - tl.max(row, axis=0)

        numerator = tl.exp(row_minus_max)

        row_sum = tl.sum(numerator, axis=0)

        softmax_output = numerator / row_sum

        output_row_start_ptr = y_ptr + idx * stride_y
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=col_mask)


def softmax(x):
    assert x.dim() == 2
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    print(x.stride(0))
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = lambda meta: (BLOCK_SIZE,)
    kernel = _softmax_kernel

    kernel[grid](x, y, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE)
    return y


def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    """
    Here is where we test the wrapper function and kernel that we wrote
    above to ensure all our values are correct, using pytorch as the
    correct answer to compare against

    we'll use an irregular number of rows & cols to verify that our padding mechanism works
    """
    # create input data
    torch.manual_seed(0)
    assert type(size) is tuple and len(size) == 2
    x = torch.randn(size[0], size[1], device=DEVICE)
    # run kernel & pytorch reference implementation
    z_tri = softmax(x)
    z_ref = torch.softmax(x, dim=1)
    # notice our implementation doesn't give a choice for what axis to softmax along.
    # this is a common theme of custom GPU kernels; because pytorch has to write code that
    #  is more general, it is slower than it could be
    # compare
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("PASSED")


if __name__ == "__main__":
    test_softmax_kernel((1025, 1025))
