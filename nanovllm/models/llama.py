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
        self.attn = Llama2Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.max_position_embeddings,
            config.head_dim,
            config.rms_norm_eps,
            config.qkv_bias,
            config.rope_theta,
            config.rope_scaling,
        )
        self.mlp = Llama2MLP()
        self.input_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope_embed = get_rope(config.hidden_size, config.max_position_embeddings)

    def forward(self,
                position_ids: torch.Tensor,
                hidden_states: torch.Tensor,
                residual: torch.Tensor
                ):
        if residual is None:
            hidden_states, residual = self.input_layer_norm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layer_norm(hidden_states, residual)
        hidden_states = self.attn(position_ids, hidden_states)
        hidden_states, residual = self.post_attn_norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Llama2MLP(nn.Module):
    pass


class Llama2Attention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 num_key_value_heads,
                 max_position_embeddings,
                 head_dim,
                 rms_norm_eps,
                 qkv_bias,
                 rope_theta,
                 rope_scaling):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.qkv_bias = qkv_bias
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_attention_heads,
            num_key_value_heads,
            bias=qkv_bias,
        )

        self.o_proj = RowParallelLinear(
            head_dim * num_attention_heads,
            hidden_size,
            qkv_bias
        )
        self.rotary_emb = get_rope(
            hidden_size,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=None,
        )
