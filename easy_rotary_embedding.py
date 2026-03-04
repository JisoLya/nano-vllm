import torch


base = 1000
rotary_dim = 10
max_position_embeddings = 10

freq = (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
inv_freq = 1.0 / freq
# 生成位置索引
t = torch.arange(max_position_embeddings, dtype=torch.float)

# print("inv_freq = \n",inv_freq)
# print("freq = \n",freq)

freqs = torch.einsum("i,j -> ij", t, inv_freq)

print(freqs.shape) # [10, 5]
cos = freqs.cos()
sin = freqs.sin()

# 左侧代表cos， 右侧代表sin
sin_cos_cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)

print(freqs.view(-1, 5, rotary_dim).shape)

print("sin_cos_cache = \n",sin_cos_cache)

