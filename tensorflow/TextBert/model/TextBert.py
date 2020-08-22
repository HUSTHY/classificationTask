import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import *
from transformers import TFBertModel


"""
这里的模型设置的是bert模型在训练的过程中不会改变权重，这个可以和bert权重参与训练做对比
"""

# class TextBert(nn.Module):
#     def __init__(self,args):
#         super(TextBert,self).__init__()
#         self.bert = BertModel.from_pretrained(args.model_path)
#         #param.requires_grad = False 训练的时候不改变初始预训练bert的权重值
#         for param in self.bert.parameters():
#             param.requires_grad = args.requires_grad
#
#         self.cl1 = nn.Linear(768,768)
#         self.dropout = nn.Dropout(0.5)
#         self.cl2 = nn.Linear(768,8)
#
#     def forward(self,input_ids,input_mask):
#         embedding = self.bert(input_ids,input_mask)[0]
#         mean_embedding = torch.mean(embedding,dim=1)
#         x = self.dropout(mean_embedding)
#         x = self.cl1(x)
#         x = self.dropout(x)
#         logit = self.cl2(x)
#         return logit

class TextBert(models.Model):
    def __init__(self,args,):
        super(TextBert,self).__init__()
        self.args = args
        self.bert = TFBertModel.from_pretrained(self.args.model_path, from_pt=True)
        # param.requires_grad = False 训练的时候不改变初始预训练bert的权重值
        self.cl1 = layers.Dense(768, activation='relu')
        self.cl2 = layers.Dense(384, activation='relu')
        self.cl3 = layers.Dense(8, activation='relu')

    # def build(self, input_shape):
    #     self.bert = TFBertModel.from_pretrained(self.args.model_path,from_pt=True)
    #     # param.requires_grad = False 训练的时候不改变初始预训练bert的权重值
    #     self.cl1 = layers.Dense(768,activation='relu')
    #     self.cl2 = layers.Dense(384, activation='relu')
    #     self.cl3 = layers.Dense(8, activation='relu')
    #     super(TextBert,self).build(input_shape)

    def call(self, inputs):
        embedding = self.bert(inputs)[0]
        mean_embedding = tf.reduce_mean(embedding,1)
        x = self.cl1(mean_embedding)
        x = tf.nn.relu(x)
        x = self.cl2(x)
        x = tf.nn.relu(x)
        logit = self.cl3(x)
        return logit





