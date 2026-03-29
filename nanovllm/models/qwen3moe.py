import torch
from torch import nn
from transformers import Qwen3MoeConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope


class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
            self,
            config: Qwen3MoeConfig
    ) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)


class Qwen3MoeModel(nn.Module):

    def __init__(
            self,
            config: Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        if (layer_idx not in config.mlp_only_layers) and (
                config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMLPBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config)

        self.attn = Qwen3MoeAttention(
            config.hidden_size,
            config.num_heads,
            config.num_key_value_heads,
            config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 1000000),
            rms_norm_eps=config.rms_norm_eps,
        )
        self.input_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor,
                residual: torch.Tensor | None = None):
        if residual is None:
            hidden_states, residual = self.input_layer_norm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layer_norm(hidden_states, residual)

        hidden_states = self.attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeAttention(nn.Module):
    def __init__(self,
                 hidden_size, num_heads, num_kv_heads, max_positions, rope_theta, rms_norm_eps):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_heads = num_kv_heads
        self.max_positions = max_positions
        self.base_freq = rope_theta
        self.rms_norm_eps = rms_norm_eps

        self.q_size = self.head_dim * self.num_heads
        self.kv_size = self.head_dim * self.num_kv_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            num_heads,
            num_kv_heads,
            False,
        )

        self.o_proj = RowParallelLinear(
            self.head_dim * num_heads,
            hidden_size,
            False,
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=self.max_positions,
            base=self.base_freq,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class Qwen3MoeSparseMLPBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(self.hidden_dim, config.num_experts)
        self.norm_topk_prob = config.norm_topk_prob
        self.num_experts = config.num_experts

        self.experts = nn.ModuleList(
            [Qwen3MoeExperts(config) for _ in range(config.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor):
        final_states = torch.zeros_like(hidden_states)

        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        logits = self.gate(hidden_states).softmax(dtype=torch.float, dim=-1)  # [seqLen, num_experts]
        router_top_value, router_top_indices = torch.topk(logits, self.top_k, dim=-1)  # [seqLen, top_k]
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)

        expert_mask = torch.nn.functional.one_hot(router_top_indices,
                                                  num_classes=self.num_experts)  # [num_tokens, top_k, num_experts]
        expert_mask = expert_mask.permute(2, 1, 0)
        active_experts = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in active_experts:
            expert_layer = self.experts[expert_idx]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, token_idx].reshape(-1, self.hidden_dim)  # [num_tokens,hidden_states]
            current_state = (
                    expert_layer(current_state) * router_top_value[token_idx, top_k_pos, None]
            )
            final_states.index_add_(
                0, token_idx, current_state.to(hidden_states.dtype)
            )
        return final_states


class Qwen3MoeExperts(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()

        self.hidden_dim = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_dim,
            [2 * self.moe_intermediate_size],
            False,
        )

        self.down_proj = RowParallelLinear(
            self.moe_intermediate_size,
            self.hidden_dim,
            False,
        )

        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor):
        gate_up = self.gate_up_proj(hidden_states)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3MoeMLP(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size * 2],
            False
        )

        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor):
        gate_up = self.gate_up_proj(hidden_states)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x
