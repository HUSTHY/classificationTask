import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

"""
这里的模型设置的是bert模型在训练的过程中不会改变权重，这个可以和bert权重参与训练做对比
"""

class TextBert(nn.Module):
    def __init__(self,args):
        super(TextBert,self).__init__()
        self.bert = BertModel.from_pretrained(args.model_path)
        #param.requires_grad = False 训练的时候不改变初始预训练bert的权重值
        for param in self.bert.parameters():
            param.requires_grad = args.requires_grad

        self.cl1 = nn.Linear(768,768)
        self.dropout = nn.Dropout(0.5)
        self.cl2 = nn.Linear(768,8)

    def forward(self,input_ids,input_mask):
        # print('input_mask', input_mask.shape)
        # import time
        # time.sleep(5000)
        embedding = self.bert(input_ids,input_mask)[0]
        mean_embedding = torch.mean(embedding,dim=1)
        x = self.dropout(mean_embedding)
        x = self.cl1(x)
        x = self.dropout(x)
        logit = self.cl2(x)
        return logit




