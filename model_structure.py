from transformers import AutoModelForCausalLM, AutoConfig
import argparse

def main(args):
    config = AutoConfig.from_pretrained(args.model_path)
    print("---模型参数--")
    print(config)

    model = AutoModelForCausalLM.from_config(config)
    print("\n--- 模型层级结构 ---")
    print(model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='/root/autodl-tmp/qwen/qwen/Qwen2.5-0.5B/')
    args = parser.parse_args()
    main(args)