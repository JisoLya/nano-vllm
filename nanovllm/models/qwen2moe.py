import torch
from torch import nn
from transformers import Qwen2MoeConfig

from nanovllm.layers.embed_head import VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.quantized_linear import GPTQLinear
from triton_impl.gptq_quantize_kernel import fused_gate_up
from triton_impl.matmul_gptq import matmul_gptq


class Qwen2MoeForCausalLM(nn.Module):
    # todo这里需要做修改
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self,
                 config: Qwen2MoeConfig):
        super().__init__()
        config.quantization_config = getattr(config, "quantization_config", None)
        self.model = Qwen2MoeModel(config)

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


class Qwen2MoeModel(nn.Module):
    def __init__(
            self,
            config: Qwen2MoeConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen2MoeDecoder(config) for _ in range(config.num_hidden_layers)
        )
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


class Qwen2MoeDecoder(nn.Module):
    def __init__(self, config: Qwen2MoeConfig) -> None:
        super().__init__()
        # moe
        self.mlp = Qwen2MoeMLPBlock(config)
        self.attn = Qwen2MoeAttention(config)
        # hidden_state = 3584 for Qwen2-57B-A14B-Instruct-GPTQ-Int4
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor,
                residual: torch.Tensor):

        if residual is None:
            residual, hidden_states = hidden_states, self.input_layernorm(hidden_states)
        else:
            # 这里都是fp16的数据格式
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        # fp16
        return hidden_states, residual


class Qwen2MoeMLPBlock(nn.Module):
    def __init__(self, config: Qwen2MoeConfig):
        super().__init__()
        self.experts = nn.ModuleList([
            Qwen2MoeExpert(config) for _ in range(config.num_experts)
        ])
        self.shared_experts = nn.ModuleList(
            Qwen2MoeExpert(config) for _ in range(config.num_shared_experts)
        )
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)
        self.hidden_dim = config.hidden_size

        self.top_k = config.num_experts_per_tok

    # todo 这一段有性能问题
    def forward(self, hidden_states: torch.Tensor):
        final_states = torch.zeros_like(hidden_states)
        shared_output = torch.zeros_like(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_dim)

        for layer in self.shared_experts:
            shared_output += layer(hidden_states)

        shared_gate_score = torch.sigmoid(self.shared_expert_gate(hidden_states))

        logits = self.gate(hidden_states).softmax(dtype=torch.float, dim=-1)
        router_top_value, router_top_indices = torch.topk(logits, self.top_k, dim=-1)
        if self.norm_topk_prob and self.top_k > 1:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)

        expert_mask = torch.nn.functional.one_hot(
            router_top_indices,
            num_classes=self.num_experts,
        )

        expert_mask = expert_mask.permute(2, 1, 0)
        active_experts = torch.greater(expert_mask.sum(dim=(-1, -2)), other=0).nonzero()

        for expert_idx in active_experts:
            expert_layer = self.experts[expert_idx]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, token_idx].reshape(-1, self.hidden_dim)
            current_state = (
                    expert_layer(current_state) * router_top_value[token_idx, top_k_pos, None]
            )
            final_states.index_add_(0, token_idx, current_state.to(hidden_states.dtype))

        final_states.add_(shared_gate_score * shared_output)
        final_down_scale = 1.0 / (self.top_k ** 0.5)
        final_states.mul_(final_down_scale)
        # fp16
        return final_states


class Qwen2MoeExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = GPTQLinear(config.hidden_size, )
        self.up_proj = GPTQLinear(config.hidden_size, )
        self.down_proj = GPTQLinear(config.hidden_size, )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = fused_gate_up(x, gate=self.gate_proj, up=self.up_proj)
        # todo 隐藏层与gptq量化的乘积， output_shape[M, hidden_size]
        x = matmul_gptq(x, self.down_proj)
        return x


# todo Attention层的dequantize直接利用flash_attn库
class Qwen2MoeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = GPTQLinear(1, config.hidden_size, config)
        self.k_proj = GPTQLinear(1, config.hidden_size, config)
        self.v_proj = GPTQLinear(1, config.hidden_size, config)
        self.o_proj = GPTQLinear(1, config.hidden_size, config)

    # 直接利用pytorch中反量化并进行flash_attn
    def forward(self, positions: torch.Tensor):
        q, k, v = self.dequantize(self.q_proj, self.k_proj, self.v_proj)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        # 需要处理一下量化后的GPTQLinear的逻辑, input()
        output = self.o_proj(o.flatten(1, -1))
        # fp16输出
        return output

    def dequantize(self, q: GPTQLinear, k: GPTQLinear, v: GPTQLinear):
        q = torch.empty()
        k = torch.empty()
        v = torch.empty()

        return q, k, v
