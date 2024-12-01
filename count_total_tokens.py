from transformers import GPT2Tokenizer
import json

# 加载 GPT-2 的分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 数据集路径（假设是一个 JSONL 文件，每行是一个样本）
data_file = "./oscar_subsets/oscar_subset_100MB_raw.jsonl"

total_tokens = 0

# 遍历数据集
with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        # 提取文本字段
        sample = json.loads(line)
        text = sample["text"]

        # 对文本进行分词
        tokens = tokenizer.encode(text, truncation=True, max_length=1024)
        total_tokens += len(tokens)

print(f"总 token 数为: {total_tokens}")
