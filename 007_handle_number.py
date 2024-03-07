import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import os
import numpy as np 

# 准备数据
BATCH_SIZE = 256
def get_dataloader(train=True):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    dataset = MNIST(root='data/', train=train, transform=transform_fn)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28*28*1, 28)
        self.fc2 = nn.Linear(28, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28*1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

model = MnistNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
if os.path.exists('./model/model.pt'):
    # model = torch.load('./model/model.pt')
    model.load_state_dict(torch.load('./model/model.pt'))
if os.path.exists('./model.optimizer.pkl'):
    # optimizer = torch.load('./model.optimizer.pkl')
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))
model.train()


def train(epoch):
    data_loader = get_dataloader()
    for idx, (input, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(input), len(data_loader.dataset),
                100. * idx / len(data_loader), loss.item()))
        
        if idx % 1000 == 0:
            torch.save(model.state_dict(), './model/model.pt')
            torch.save(optimizer.state_dict(), './model/optimizer.pkl')

def test():
    model.eval()
    loss_list = []
    acc_list = []
    test_dataloader = get_dataloader(train=False)
    for idx, (input, target) in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output, target)
            loss_list.append(cur_loss.item())
            # 计算准确率
            pred = output.max(dim=-1)[-1]
            cur = pred.eq(target).float().mean()
            acc_list.append(cur.item())
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(np.mean(loss_list), np.mean(acc_list)))



if  __name__ == '__main__':
    """     for epoch in range(1, 5):
        train(epoch) """
    test()