import torch
import torch.nn as nn
from torch.optim import SGD

# 准备数据
x = torch.rand([500, 1])
y_true = 3 * x + 0.8

# 定义模型
class MyLiner(nn.Module):
    def __init__(self):
        super(MyLiner, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# 实例化模型、优化器、和损失函数
model = MyLiner()
optimizer = SGD(model.parameters(), lr = 0.01)
criteria = nn.MSELoss()

# 循环进行梯度下降、梯度的更新
for i in range(10000):
    # 预测
    y_predict = model(x)
    # 计算loss
    loss = criteria(y_predict, y_true)
    # 清空梯度
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()
    if(i % 100 == 0):
        print(f'第{i}次训练，loss={loss.item()}， params={model.linear.weight.item(), model.linear.bias.item()}')