import deepspeed
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        print("Not running in distributed mode.")

class DummyDataset(Dataset):
    def __init__(self, tokenizer, size=100, max_len=128):
        self.tokenizer = tokenizer
        self.size = size
        self.max_len = max_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_text = "This is a dummy sentence for testing DeepSpeed."
        inputs = self.tokenizer(
            input_text,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        label = torch.tensor(0)  # 假设二分类任务，标签为0或1
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": label,
        }


def create_model_and_optimizer():
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    optimizer = Adam(params=model.parameters(), lr=1e-5)
    return model, optimizer


def main():
    init_distributed()
    print(f"Running on rank {dist.get_rank()} of {dist.get_world_size()}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = DummyDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model, optimizer = create_model_and_optimizer()

    # DeepSpeed 配置
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 10,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 1e-5},
        },
        "fp16": {
            "enabled": True,  # 如果有A100或V100 GPU，建议启用FP16
        },
    }

    # 初始化DeepSpeed引擎
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=deepspeed_config
    )

    # 模拟训练
    model_engine.train()
    for epoch in range(3):
        for step, batch in enumerate(dataloader):
            inputs = {
                "input_ids": batch["input_ids"].to(model_engine.local_rank),
                "attention_mask": batch["attention_mask"].to(model_engine.local_rank),
                "labels": batch["labels"].to(model_engine.local_rank),
            }
            outputs = model_engine(**inputs)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
