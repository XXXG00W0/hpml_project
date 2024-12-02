import os, torch
from datasets import load_dataset
from transformers import GPT2Tokenizer

# 设置存储目录（HPC目录）
output_dir = "./oscar2301/"  # 替换为实际的HPC目录路径
os.makedirs(output_dir, exist_ok=True)

# 创建输出文件
output_file_txt = os.path.join(output_dir, "oscar_2301_200mb.txt")
if not os.path.exists(output_file_txt):

    # 加载 OSCAR 23.01 的特定子集（以英语为例）
    subset_size = 200 * 1024 * 1024  # 200MB
    dataset = load_dataset("oscar-corpus/OSCAR-2301", "en", split="train", streaming=True)


    # 下载并存储
    with open(output_file_txt, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example["text"].strip()
            if text:
                f.write(text + " <|endoftext|>\n")  # 每段文本加上 GPT-2 分隔符

    print(f"Sample of 200MB from OSCAR 23.01 downloaded to {output_file_txt}")
else:
    print(f"Sample of 200MB from OSCAR 23.01 already exists at {output_file_txt}")

# 转换为 PyTorch 格式
output_file_pt = os.path.join(output_dir, "oscar_2301_200mb.pt")
if os.path.exists(output_file_pt):
    print(f"Sample of 200MB from OSCAR 23.01 already converted to {output_file_pt}")
    exit()
else:
    print(f"Converting sample of 200MB from OSCAR 23.01 to PyTorch format...")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# 设置 pad_token 为 eos_token
tokenizer.pad_token = tokenizer.eos_token

with open(output_file_txt, "r", encoding="utf-8") as f:
    for line in f:
        tokens = tokenizer(line, return_tensors="pt", truncation=True, padding=True)
        with open(output_file_pt, "ab") as f_out:
            torch.save(tokens, f_out)

print(f"Sample of 200MB from OSCAR 23.01 downloaded and converted to {output_file_pt}")
