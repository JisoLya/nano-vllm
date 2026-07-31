import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file


# --- 1. 定义你的 GPTQLinear 类 ---
class GPTQLinear(nn.Module):
    def __init__(self, input_size, output_size, bits=4, group_size=128):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bits = bits
        self.group_size = group_size

        # 定义容器 (物理尺寸)
        pack_factor = 32 // bits
        self.qweight = nn.Parameter(
            torch.empty((input_size // pack_factor, output_size), dtype=torch.int32),
            requires_grad=False
        )
        self.qzeros = nn.Parameter(
            torch.empty((input_size // group_size, output_size // pack_factor), dtype=torch.int32),
            requires_grad=False
        )
        self.scales = nn.Parameter(
            torch.empty((input_size // group_size, output_size), dtype=torch.float16),
            requires_grad=False
        )


# --- 2. 模拟创建一个 safetensors 文件 (模拟下载的模型) ---
def create_mock_safetensors(path):
    # 模拟 k_proj 的数据
    # input=3584, output=512
    tensors = {
        "model.layers.0.self_attn.k_proj.qweight": torch.randint(0, 100, (448, 512), dtype=torch.int32),
        "model.layers.0.self_attn.k_proj.qzeros": torch.randint(0, 100, (28, 64), dtype=torch.int32),
        "model.layers.0.self_attn.k_proj.scales": torch.randn((28, 512), dtype=torch.float16),
        "model.layers.0.self_attn.k_proj.bias": torch.randn((512,), dtype=torch.float16),  # 即使模型有偏置
    }
    save_file(tensors, path)
    print(f"成功创建模拟文件: {path}")


# --- 3. 编写加载逻辑 ---
def load_gptq_weights(model_layer, tensors, prefix):
    """
    model_layer: GPTQLinear 实例
    tensors: load_file 读出来的字典
    prefix: 该层在文件里的前缀，例如 'model.layers.0.self_attn.k_proj'
    """
    # 建立映射：文件里的后缀 -> 模型里的属性名
    mapping = {
        "qweight": "qweight",
        "qzeros": "qzeros",
        "scales": "scales"  # 注意这里对齐了你之前的 scales 命名
    }

    for suffix, attr_name in mapping.items():
        full_key = f"{prefix}.{suffix}"
        if full_key in tensors:
            new_data = tensors[full_key]
            param = getattr(model_layer, attr_name)

            # 关键：检查维度是否匹配
            if param.shape == new_data.shape:
                param.data.copy_(new_data)  # 使用 copy_ 减少显存碎片
                print(f"  [成功] 加载 {full_key} -> {attr_name}")
            else:
                print(f"  [失败] 维度不匹配: {full_key} {new_data.shape} vs {param.shape}")


# --- 4. 运行 Demo ---
if __name__ == "__main__":
    file_path = "demo_qwen2_k_proj.safetensors"
    # create_mock_safetensors(file_path)

    # 1. 初始化模型层 (3584 -> 512)
    k_proj = GPTQLinear(3584, 512)

    # 2. 读取文件
    state_dict = load_file(file_path)

    # 3. 执行加载
    print("\n开始执行加载流程:")
    load_gptq_weights(k_proj, state_dict, "model.layers.0.self_attn.k_proj")
