import os
from huggingface_hub import snapshot_download

# 设置环境变量加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def download():
    repo_id = "Qwen/Qwen2.5-0.5B"
    local_dir = "/root/autodl-tmp/Qwen2.5-0.5B"

    print(f"开始下载模型 {repo_id} 到 {local_dir}...")

    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=False  # 如果是公开模型不需要token
        )
        print(f"下载成功！模型路径: {path}")
    except Exception as e:
        print(f"下载失败: {e}")


if __name__ == "__main__":
    download()