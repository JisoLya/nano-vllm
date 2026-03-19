import argparse

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def main(arg):
    tokenizer = AutoTokenizer.from_pretrained(arg.model_path)
    llm = LLM(arg.model_path, enforce_eager=True, tensor_parallel_size=arg.tp_size)

    sampling_params = SamplingParams(temperature=arg.temperature, max_tokens=arg.max_tokens)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    # 模型入口
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=1)
    # model path
    parser.add_argument("--model-path", type=str, default='/root/autodl-tmp/qwen/Qwen3-0.5B/Qwen/Qwen3-0.6B/')
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    main(args)
