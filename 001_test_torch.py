import torch
import numpy as np

t1 = torch.Tensor([1, 2, 3])
t2 = torch.Tensor([4, 5, 6])
t3 = t1 + t2
print(t3)

n1 = np.arange(12).reshape(3, 4)
print(n1)

print(torch.Tensor(n1))

print(torch.empty(3, 4))
print(torch.ones([3, 4]))
print(torch.zeros([3, 4]))
print(torch.rand([3, 4]))
print(torch.eye(3))
print(torch.arange(0, 10, 2))
print(torch.linspace(0, 10, 6))
print(torch.rand([3, 4, 6]))
print(torch.randint(low=3, high=10, size=[3, 4]))
print(torch.randn([3, 4]))