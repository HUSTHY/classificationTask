import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import *
from transformers import TFBertModel

"""
这里的模型设置的是bert模型在训练的过程中不会改变权重，这个可以和bert权重参与训练做对比
"""

class TextBert(models.Model):
    def __init__(self,args,):
        super(TextBert,self).__init__()
        self.args = args
        #由于没有下载到model.h5模型权重，这里得使用pytorch_model.bin格式文件加载from_pt = true
        self.bert = TFBertModel.from_pretrained(self.args.model_path, from_pt=True)
        # param.requires_grad = False 训练的时候不改变初始预训练bert的权重值
        self.cl1 = layers.Dense(768, activation='relu')
        self.cl2 = layers.Dense(384, activation='relu')
        self.cl3 = layers.Dense(8, activation='softmax')


    def call(self, inputs):
        embedding = self.bert(inputs)[0]
        mean_embedding = tf.reduce_mean(embedding,1)
        x = self.cl1(mean_embedding)
        x = tf.nn.relu(x)
        x = self.cl2(x)
        x = tf.nn.relu(x)
        logit = self.cl3(x)
        return logit





