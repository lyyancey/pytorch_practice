"""
构建词典、把句子转化为序列
"""

class Word2Sequence:
    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'

    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}
    
    def fit(self, sentence):
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1
    
    def build_vocab(self, min=5,max=None, max_features=None):
        """
        生成词典
        :param min: 最小词频
        :param max: 最大词频
        :param max_features: 最多要保留多少个词语
        """
        # self.count = {word: value for word,value in self.count if (min is None or value >= min) and (max is None or value <=max)}
        if min is not None:
            self.count = {word: value for word, value in self.count.items() if value >= min}
        if max is not None:
            self.count = {word: value for word, value in self.count.items() if value <= max}
        # 限制保留的词语数量
        if max_features is not None:
            temp = sorted(self.count.items(), key=lambda x:x[-1], reverse=True)[:max_features]
            self.count = dict(temp)
        
        for word in self.count:
            self.dict[word] = len(self.dict)
        
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        """
        把句子转化为序列
        :param sentence: 句子
        :param max_len: 最大长度
        :return: 返回句子对应的序列
        """
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG] * (max_len -len(sentence))
            elif max_len < len(sentence):
                sentence = sentence[:max_len]
        return [self.dict.get(word, self.UNK) for word in sentence]
    
    def inverse_transform(self, sequence):
        """
        把序列转化为句子
        :param sequence: 序列
        :return: 返回序列对应的句子
        """
        return [self.inverse_dict.get(idx) for idx in sequence]
    
    def __len__(self):
        return len(self.dict)

if __name__ == '__main__':
    
    from word_sequence import Word2Sequence
    import pickle
    import os
    from dataset import tokenizer
    from tqdm import tqdm

    ws = Word2Sequence()
    data_paths  = r'../data/aclImdb/train'
    temp_data_path = [os.path.join(data_paths, 'pos'), os.path.join(data_paths,'neg')]
    for data_path in temp_data_path:
        file_name_list = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if file_name.endswith('.txt')]
        for file_path in tqdm(file_name_list):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                content = tokenizer(content)
                ws.fit(content)
    ws.build_vocab(min=10, max_features=10000)
    pickle.dump(ws, open(r'../model/ws.pkl', 'wb'))
    print(ws.dict)
    print(len(ws.dict))
