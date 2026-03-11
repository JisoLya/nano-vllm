import torch

a = torch.Tensor([
    [2, 3, 4, 1, 5, 7],
    [1, 2, 3, 4, 0, 6],
])

b, c = torch.topk(a, 2, dim=-1)
#
one_hot = torch.nn.functional.one_hot(
    c, 6
)
# (x, y ,z) 第x个token， y: 排名分别为第一、第二、第三的专家位置,z专家个数
print(c)

print(one_hot[0][0][1])

print(one_hot.shape)
