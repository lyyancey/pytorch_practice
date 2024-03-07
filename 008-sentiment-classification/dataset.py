from torch.utils.data import DataLoader, Dataset
import os
import re
from lib import ws, max_len
import torch
def tokenizer(content):
    content = re.sub("<.*?>", " ", content)
    """  filters = ['!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.',
            '/', ':', ';', '=', '>', '?', '@', '[', '\\', ']', '^', '_',
            '`', '{', '|', '}', '~', '\t', '\n', '\x97', '\x96', '”', '“', '’', '‘', '—'] """
    filters = r'[\!"#$%&\()*+,\-./:;<=>?@\[\\\]^_`{|}~\t\n—“”‘’]'
    content = re.sub(filters, " ", content)
    tokens = [i.strip() for i in content.lower().split()]
    return tokens

class IMDBDataset(Dataset):
    def __init__(self, train=True):
        self.train_data_path = r'../data/aclImdb/train'
        self.test_data_path = r'../data/aclImdb/test'

        data_path = self.train_data_path if train else self.test_data_path
        self.total_file_path = []
        temp_file_path = [os.path.join(data_path, 'pos'), os.path.join(data_path, 'neg')]
        # 把所有的文件名放入列表
        for path in temp_file_path:
            file_names = os.listdir(path)
            file_path_list = [os.path.join(path, file_name) for file_name in file_names if file_name.endswith('.txt')]
            self.total_file_path.extend(file_path_list)

    def __getitem__(self, index):
        file_path = self.total_file_path[index]
        # 获取label
        lable = file_path.split('\\')[-2]
        label = 1 if lable == 'pos' else 0
        # 获取文件内容
        content = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            content = tokenizer(content)
        return content, label

    def __len__(self):
        return len(self.total_file_path)


def collate_fn(batch):
    '''
    :param batch: ([tokens, label], [tokens, label])    
    '''
    content, label = list(zip(*batch))
    content = [ws.transform(i, max_len=max_len) for i in content]
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)
    return content, label


def get_dataloader(train=True, batch_size=2):
    imdb_dataset = IMDBDataset(train)
    data_loader = DataLoader(imdb_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return data_loader

if __name__ == '__main__':
    for idx, (content, target) in enumerate(get_dataloader()):
        print(idx)
        print(content)
        print(target)
        break
