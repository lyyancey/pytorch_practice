import torch.nn as nn
import lib
from  torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import torch
from dataset import IMDBDataset, get_dataloader
import os
import numpy as np

class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(len(lib.ws), 100)
        # LSTM
        self.lstm = nn.LSTM(
            input_size=100,
            hidden_size=lib.hidden_size,
            num_layers=lib.num_layers,
            bidirectional=lib.bidirectional,
            batch_first=True, 
            dropout=lib.dropout) 
        self.layer = nn.Sequential(
            nn.Linear(lib.hidden_size*2, lib.hidden_size),
            nn.ReLU(True),
            nn.BatchNorm1d(lib.hidden_size),
            nn.Dropout(0.5),
            nn.Linear(lib.hidden_size, lib.hidden_size),
            nn.ReLU(True),
            nn.BatchNorm1d(lib.hidden_size),
            nn.Dropout(0.5),
            nn.Linear(lib.hidden_size, 2)
        )
        # self.fc = nn.Linear(lib.hidden_size*2, 2)
    
    def forward(self, x):
        """
        input: x,shape = (batch_size, max_len)
        """
        x = self.embedding(x) # 进行embedding操作 shape = (batch_size, max_len, embed_dim) 
        x, (h_n, c_n) = self.lstm(x) # x shape = (batch_size, max_len, hidden_size*2)  h_n shape (2*2, batch_size, hidden_size)
        # 获取两个方向最后一次的output,进行concat操作
        output_fw = h_n[-2, :, :]
        output_bw = h_n[-1, :, :]
        output = torch.cat([output_fw, output_bw], dim=-1) # [batch_size, hidden_size * 2]

        # x = output.view([-1, lib.max_len*100])
        # out = self.fc(output)
        out = self.layer(output)
        return F.log_softmax(out, dim=-1)


model = SentimentClassifier().to(lib.device)
optimizer = Adam(model.parameters(), lr=0.001)
if os.path.exists('./model/model.pkl'):
    model.load_state_dict(torch.load('./model/model.pkl'))

if os.path.exists('./model/optimizer.pkl'):
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))

dataloader = get_dataloader(batch_size=128)
def train(epoch):
    for idx, (input, target) in enumerate(dataloader):
        input = input.to(lib.device)
        target = target.to(lib.device)
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('epoch: {}, idx: {}, loss: {}'.format(epoch, idx, loss.item()))
        if idx % 100 == 0:
            torch.save(model.state_dict(), './model/model.pkl')
            torch.save(optimizer.state_dict(), './model/optimizer.pkl')

def eval():
    loss_list = []
    acc_list = []
    for idx, (input, target) in tqdm(enumerate(get_dataloader(train=False)), total=len(get_dataloader(train=False)), ascii=True, desc='eval'):
        input = input.to(lib.device)
        target = target.to(lib.device)
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output, target)
            loss_list.append(cur_loss.cpu().item())
            # 计算准确率
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc.cpu().item())
    print('idx: {}, loss:{}, acc:{}, avg_loss:{}, avg_acc:{}'.format(idx, cur_loss, cur_acc, np.mean(loss_list), np.mean(acc_list)))



if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
    eval()