from torch.utils.data import Dataset
import torch
from gensim.models import Word2Vec
import pandas as pd
from tqdm import tqdm

class ReadDataSet(Dataset):
    def __init__(self,file_path,repeat=1):
        self.repeat = repeat
        self.wv_model = Word2Vec.load('pretrain_model/word2vec/word2vec_newtrain.model')
        self.wv_words = self.wv_model.wv.index2word
        self.wv_dim = 100
        self.data_list = self.read_file(file_path)
        self.output = self.word2vec_paddings_tensor(self.data_list)

    def read_file(self,file_path):
        data_list = []
        df = pd.read_csv(file_path) # tsv文件
        texts, labels = df['text_padding'], df['label']
        for text, label in tqdm(zip(texts, labels),desc='read data from csv files:'):
            text = text.split(' ')[0:50]
            data_list.append((text,label))
        return data_list

    def word2vec_paddings_tensor(self,data_list):
        output = []
        for data,label in tqdm(data_list,desc='text to vord2vec:'):
            vec = []
            for word in data:
                v = self.wv_model[word].tolist()
                vec.append(v)
            vec = torch.tensor(vec)
            label = torch.tensor(int(label))
            res = (vec,label)
            output.append(res)
        return output

    def __getitem__(self, item):
        text = self.output[item][0]
        label = self.output[item][1]
        return text,label

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.output)
        return data_len


