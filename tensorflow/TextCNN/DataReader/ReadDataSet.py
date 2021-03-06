import tensorflow as tf
from gensim.models import Word2Vec
import pandas as pd
from tqdm import tqdm
import numpy as np

class ReadDataSet():
    def __init__(self,file_path,repeat=1):
        self.max_sentence_length = 200
        self.repeat = repeat
        self.wv_model = Word2Vec.load('pretrain_model/word2vec/word2vec_newtrain.model')
        self.wv_words = self.wv_model.wv.index2word
        self.wv_dim = 100
        self.data_list = self.read_file(file_path)
        self.output = self.word2vec_paddings_tensor(self.data_list)
        if 'trian' in file_path:
            self.len_train = len(self.data_list)

    def read_file(self,file_path):
        data_list = []
        df = pd.read_csv(file_path)# tsv文件
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
            vec = tf.constant(vec)
            label = tf.constant(int(label))
            res = (vec,label)
            output.append(res)
        return output

    def __call__(self, *args, **kwargs):
        for features,labels in self.output:
            yield features,labels
