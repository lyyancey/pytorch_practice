import torch
from torch.utils.data import Dataset, DataLoader
data_path = r'./data/SMSSpamCollection'

# 实现一个自定义的数据集
class CustomDataset(Dataset):
    def __init__(self):
        self.lines = open(data_path, 'r', encoding='utf-8').readlines()


    # 获取索引对应位置的一条数据
    def __getitem__(self, index):
        cur_line = self.lines[index].strip()
        label = cur_line[:4].strip()
        content = cur_line[4:].strip()
        return label, content

    # 获取数据集的长度
    def __len__(self):
        return len(self.lines)

myDataset = CustomDataset()
dataLoader = DataLoader(myDataset, batch_size=7, shuffle=True, num_workers=0,drop_last=True)

if __name__ == '__main__':
    # data_set = CustomDataset()
    # print(data_set[0])
    # print(len(data_set))
    for i, (label, content) in enumerate(dataLoader):
        print(i)
        print(label)
        print(content)
        print(len(dataLoader))
        print(len(myDataset))
        # print()
        # print(label)
        # print(content)
        # print('-------------------')
        # if i > 5:
        #     break
        break