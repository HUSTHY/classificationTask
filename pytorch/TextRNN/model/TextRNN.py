import torch
import torch.nn as nn


"""
TextRNN,其实就是利用了Bilstm把句子的最后时刻或者说是最后那个字（这里可能不好理解）的hidden state，拿出来喂入分类器中，进行分类的。
这里仍然没有使用随机的embedding，我们仍然使用word2vec的词向量，经过操作来生成文本向量。
开始hidden_size设置为200，发现效果太差了，loss都不下降的
50词语的时候验证集准确率能到73%

训练过程中还是要监控验证集准确率


"""

class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN,self).__init__()
        self.embedding_dim = 100 #文本词或者字的向量维度
        self.hidden_size = 50 #lstm的长度，可以和seq_legth一样，也可以比它长
        self.layer_num = 2
        self.class_num = 8


        self.lstm = nn.LSTM(self.embedding_dim, # x的特征维度,即embedding_dim
                            self.hidden_size, # stm的长度，可以和seq_legth一样，也可以比它长
                            self.layer_num, # 把lstm作为一个整体，然后堆叠的个数的含义
                            batch_first=True,
                            bidirectional=True
                            )
        self.classificer = nn.Linear(self.hidden_size*self.layer_num,self.class_num)

    def forward(self,x):
        #x的维度为(batch_size, time_step, input_size=embedding_dim)

        # 隐层初始化
        # h0维度为(num_layers*direction_num, batch_size, hidden_size)
        # c0维度为(num_layers*direction_num, batch_size, hidden_size)
        h0 = torch.zeros(self.layer_num*2,x.size(0),self.hidden_size).to('cuda')
        c0 = torch.zeros(self.layer_num*2,x.size(0),self.hidden_size).to('cuda')

        #out维度为(batch_size, seq_length, hidden_size * direction_num)
        out,(hn,cn)  =self.lstm(x,(h0,c0))
        #最后一步的输出, 即(batch_size, -1, output_size)
        logit = self.classificer(out[:,-1,:])  # (B, C)
        return logit




