import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # temperature 在unsqueeze前的形状为[batch_size], unsqueeze后的形状变为[batch_size,1]
        # logits的形状为[batch_size, vocab_size]， 直接广播，重新
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # 绝对的贪婪采样
        # sample_tokens = logits.argmax(dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        # argmax直接会返回当前行的最大值下标(也就是token_id)，这样就得到了[batch_size]形状的sample_tokens
        return sample_tokens
