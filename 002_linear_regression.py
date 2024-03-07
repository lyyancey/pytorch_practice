import torch
from torch import nn
import matplotlib.pyplot as plt

# 学习率
lr_rate = 0.01
# 1. 准备数据
# y = 3*x + 0.8
x = torch.rand([500, 1])
y_true = 3 * x + 0.8

# 通过模型计算y_predict
w = torch.rand([1, 1], requires_grad=True)
b = torch.tensor(0,dtype=torch.float, requires_grad=True)

class LG(nn.Module):
    def __init__(self):
        super(LG, self).__init__()
        self.liner = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.liner(x)


model = LG()
criteria = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

# 通过模型、反向传播, 更新参数
for i in range(10000):
    
    # 预测
    # y_predict = torch.matmul(x, w) + b
    y_predict = model(x)
    # 计算loss
    #  loss = (y_true - y_predict).pow(2).mean()
    loss = criteria(y_true, y_predict)

    """     if w.grad is not None:
        w.grad.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_() """
    # 清空梯度
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新参数
    """     w.data = w.data - lr_rate * w.grad
    b.data = b.data - lr_rate * b.grad """
    optimizer.step()
    print(f'第{i}次训练，loss={loss.item()}')
    print(f'w={w.item()}, b={b.item()}')

""" plt.figure(figsize=(20, 8))
plt.scatter(x.numpy().reshape(-1), y_true.numpy().reshape(-1), c='r')

y_predict = torch.matmul(x, w) + b
plt.plot(x.numpy().reshape(-1), y_predict.detach().numpy().reshape(-1), c='r')

plt.show() """

# 4. 模型评估
model.eval()
predict = model(x)
predict = predict.detach().numpy()
plt.scatter(x.data.numpy(), y_true.data.numpy(), c='r')
plt.plot(x.data.numpy(), predict, c='b')
plt.show()