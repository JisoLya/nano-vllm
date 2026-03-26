import torch
from torch import nn
from nanovllm.layers.linear import LinearBase, ColumnParallelLinear


class GPTQLinear(LinearBase):
    def __init__(self, input_size, output_size, config):
        super(LinearBase, self).__init__()

        self.bits = config.quantization_config["bits"]  # 4
        self.group_size = config.quantization_config["group_size"]  # 128

        self.qweight = nn.Parameter(
            torch.empty((input_size // (32 // self.bits), output_size), dtype=torch.int32),
        )

        # 零点个数，由于group_size 是128
        # input_size int4 的个数， group多少个int4共享一个零点，存储进int32
        self.qzeros = nn.Parameter(
            torch.empty((input_size // self.group_size, output_size // (32 // self.bits)), dtype=torch.int32)
        )

        self.scales = nn.Parameter(
            torch.empty((input_size // self.group_size, output_size), dtype=torch.float16)
        )
        self.bias = nn.Parameter(
            torch.empty((output_size,), dtype=torch.float16),
            requires_grad=False
        )
        self.g_idx = nn.Parameter(
            torch.empty((input_size,), dtype=torch.int32),
        )

        for p in [self.qweight, self.qzeros, self.scales, self.bias]:
            p.weight_loader = self.weight_loader
        self.g_idx.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = None):
        if shard_id is None:
            param.data.copy_(loaded_weight)
            return

        output_dim = param.data.shape[-1]
        shard_size = output_dim // 2

        start_idx = shard_id * shard_size
        end_idx = (shard_id + 1) * shard_size

        param.data[..., start_idx:end_idx].copy_(loaded_weight)
        # print(f"  [Merge] Loaded {shard_id} into range [{start_idx}:{end_idx}]")

    def forward(self, x):
        # 暂时留空，之后我们需要在这里集成 Marlin 或 ExLlamaV2 的 C++ 算子
        raise NotImplementedError("需要集成 4-bit MatMul Kernel")


"""
@DeprecationWarning
class GPTQMergedColumnParallelLinear(GPTQLinear):

    def __init__(
            self,
            input_size: int,
            output_sizes: list[int],
            bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 将分配的存储空间进行裁剪
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
"""
