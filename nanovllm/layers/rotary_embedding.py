from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # 旋转的基频， 由于奇数列和偶数列共享同一个频率，只是正余弦的区别
        # 这里的base是10000
        self.inv_freq = nn.Parameter(torch.Tensor(64))
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # max_position_embeddings 模型支持的最大上下文长度，生成位置索引[0, 1,...., max_position_embeddings]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 将两个向量相乘， t[i]乘以inv_freq[j]作为得到矩阵中位置[i,j]的元素
        # 简单的例子可以查看easy_rotary_embedding.py
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # 一个Token的Query向量可能有128维。RoPE把这128维分成 64组，每组2个分量（之前提到的i与i+1共享一个频率, i = 0,2,4...）
        # freq的行代表了处理第几个token，这个token嵌入的维度需要做多少旋转
        cos = freqs.cos()
        sin = freqs.sin()
        # 为什么需要扩展维度，便于pytorch的广播因为qk的维度是[num_tokens, num_heads, head_dim]
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
        rope_scaling=None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
