from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights # 需要安装 accelerate: pip install accelerate
import argparse

def main(args):
    # 1. 仅加载配置文件（内存占用极小）
    config = AutoConfig.from_pretrained(args.model_path)
    print("--- 模型参数 ---")
    print(config)

    # 2. 在 meta device 上初始化模型结构
    print("\n--- 模型层级结构 ---")
    with init_empty_weights():
        # 这里不会分配实际内存，仅构建图结构
        model = AutoModelForCausalLM.from_config(config)
        print(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 确保路径指向 Llama-7B 或类似的目录
    parser.add_argument("--model-path", type=str, default='/root/autodl-tmp/qwen/qwen/Qwen2.5-0.5B/')
    args = parser.parse_args()
    main(args)