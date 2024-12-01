import os
from datasets import load_dataset

# 设置存储目录（HPC目录）
hpc_directory = "./oscar2301/"  # 替换为实际的HPC目录路径
os.makedirs(hpc_directory, exist_ok=True)

# 加载 OSCAR 23.01 的特定子集（以英语为例）
subset_size = 200 * 1024 * 1024  # 200MB
# dataset = load_dataset("oscar-corpus/OSCAR-2301", "en", split="train", streaming=True)
dataset = load_dataset("oscar-corpus/OSCAR-2301", "en", split="train")

# 创建输出文件
output_file = os.path.join(hpc_directory, "oscar_2301_200mb.txt")

# 下载并存储
with open(output_file, "w", encoding="utf-8") as f:
    total_size = 0
    for example in dataset:
        text = example["text"]
        f.write(text + "\n")
        total_size += len(text.encode("utf-8"))
        if total_size >= subset_size:
            break

print(f"Sample of 200MB from OSCAR 23.01 downloaded to {output_file}")
