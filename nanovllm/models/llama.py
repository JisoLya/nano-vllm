from typing import Any

import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers import LlamaConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Llama2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
            self,
            config: LlamaConfig
    ) -> None:
        super().__init__()
        # 推理
        self.model = Llama2Model(config)
        # 计算概率
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, position_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


class Llama2Model(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        # 初始化DecoderLayer
        self.layers = nn.ModuleList(
            [
                Llama2DecoderLayer(config) for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor):
        # 获得词嵌入
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(position_ids, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Llama2DecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.self_attn = Llama2Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.max_position_embeddings,
            getattr(config, "head_dim", None),
            config.rms_norm_eps,
            getattr(config, "rope_theta", 1000000),
            # config.rope_scaling,
        )
        self.mlp = Llama2MLP(
            config.intermediate_size,
            config.hidden_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
                position_ids: torch.Tensor,
                hidden_states: torch.Tensor,
                residual: torch.Tensor
                ):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(position_ids, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Llama2MLP(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Llama2Attention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 num_key_value_heads,
                 max_position_embeddings,
                 head_dim: int | None = None,
                 rms_norm_eps: float = 1e-06,
                 qkv_bias: bool = False,
                 rope_theta: float = 10000,
                 rope_scaling: tuple | None = None):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.q_size = head_dim * num_attention_heads
        self.kv_size = head_dim * num_key_value_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            num_attention_heads,
            num_key_value_heads,
            bias=False,
        )

        self.o_proj = RowParallelLinear(
            self.q_size,
            hidden_size,
            False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=None,
        )
        self.attn = Attention(
            num_heads=num_attention_heads,
            head_dim=head_dim,
            scale=rope_scaling,
            num_kv_heads=self.num_key_value_heads,
        )

    def forward(self, position_ids, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_attention_heads, self.head_dim)
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)

        q, k = self.rotary_emb(position_ids, q, k)
        o = self.attn(q, k, v)
        # o.shape [num_tokens, head_num, head_dim]
        # 需要将o展平为[num_tokens, hidden_size]
        output = self.o_proj(o.flatten(1, -1))
        return output
