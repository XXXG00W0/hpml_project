import horovod.torch as hvd
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    # 初始化 Horovod
    hvd.init()

    # 获取当前进程的 rank 和总进程数
    rank = hvd.rank()
    size = hvd.size()
    print(f"Hello from rank {rank} out of {size} processes.")

    # 在每个进程中设置 GPU 设备（如果可用）
    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 创建一个简单的模型
    model = nn.Linear(10, 1).to(device)

    # 定义一个简单的优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 将优化器封装为 Horovod 的分布式优化器
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # 广播模型的初始参数到所有进程
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # 创建一个随机输入张量
    inputs = torch.randn(64, 10).to(device)
    targets = torch.randn(64, 1).to(device)

    # 定义损失函数
    loss_fn = nn.MSELoss()

    # 模拟一个训练步骤
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        if rank == 0:  # 仅主进程输出结果
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

if __name__ == "__main__":
    main()
