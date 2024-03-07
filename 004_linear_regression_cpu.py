import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import time

# 定义数据
x = torch.rand([500, 1])
y = 3 * x + 0.8
# 定义模型
class MyLinear(nn.Module):
    def __init__(self):
        super(MyLinear, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
    
# 实例化模型、loss、optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x, y = x.to(device), y.to(device)
model = MyLinear().to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# 训练模型

for i in range(3000):
    # forward
    out = model(x)
    # 计算loss
    loss = criterion(out, y)
    # 参数梯度清零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 参数更新
    optimizer.step()
    if (i + 1) % 100 == 0:
        params = list(model.parameters())
        print(f'第{i+1}次训练后的参数w: {params[0].item()}, b: {params[1].item()}, loss: {loss.item()}')