import torch
import torch.nn as nn

# 为了 Debug 方便，固定随机种子
torch.manual_seed(42)

# 1. 模拟输入数据：2 个 Token，每个 Token 维度为 4
# [num_tokens, hidden_dim]
hidden_states = torch.Tensor([
    [2, 1, 6, 7],  # Token 0
    [0, 3, 2, 6],  # Token 1
])

# 2. 定义 Router (Gate)：4 个专家
# 权重形状为 [4, 4]，即 [num_experts, hidden_dim]
gate = nn.Linear(4, 4, bias=False)

# 3. 计算 Logits (专家得分)
logits = gate(hidden_states)
print("--- 1. Logits (每个 Token 对 4 个专家的原始得分) ---")
print(f"Shape: {logits.shape}")  # [2, 4]
print(logits)
print()

# 4. Top-K 筛选：每个 Token 选出最强的 3 个专家
# top_k = 3
router_top_value, router_top_indices = torch.topk(logits, 2, dim=-1)

print("--- 2. Top-K 结果 ---")
print(f"最强得分 (Values):\n{router_top_value}")  # 每个 Token 选出的 3 个最高分
print(f"对应专家索引 (Indices):\n{router_top_indices}")  # 每个 Token 选出的 3 个专家 ID
print()

# 5. 生成 One-Hot 掩码
# 这一步生成的形状是 [num_tokens, top_k, num_experts] -> [2, 3, 4]
expert_mask = torch.nn.functional.one_hot(router_top_indices, num_classes=4)

print("--- 3. One-Hot Mask (初步生成) ---")
print(f"Shape: {expert_mask.shape}")  # [2, 3, 4]
# 它表示：对于第 n 个 Token 的第 k 个选择，是否选中了第 e 个专家
print(expert_mask)
print()

# 6. 维度置换 (至关重要的一步，对应你看到的源码逻辑)
# 将形状转为 [num_experts, top_k, num_tokens] -> [4, 3, 2]
# 这样我们可以“站在专家的视角”看问题
expert_mask_permuted = expert_mask.permute(2, 1, 0)

print("--- 4. Permuted Mask (专家视角) ---")
print(f"Shape: {expert_mask_permuted.shape}")  # [4, 3, 2]
print()

# 7. 统计每个专家的工作量并筛选活跃专家
# sum(dim=(-1, -2)) 对最后两个维度（top_k 和 num_tokens）求和
expert_workload = expert_mask_permuted.sum(dim=(-1, -2))
# nonzero() 找出索引，flatten() 铺平
active_experts = torch.greater(expert_workload, 0).nonzero().flatten()

for expert_idx in active_experts:
    print(expert_idx)

print(f"\n最终被点名的活跃专家 ID: {active_experts}")

print(hidden_states[None, [0, 1]].reshape((-1, 4)).shape)
