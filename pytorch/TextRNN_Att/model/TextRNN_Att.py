import torch
import torch.nn as nn
import torch.nn.functional as F



"""
这里需要实现一个attention模块，这里就是用一般的attention，而不是特殊的self-attention机制等
attention的一种公式:
M = tanh(H)
a = softmax(WM)
att_score = H*a
上面的是矩阵形式
"""

class TextRNN_Att(nn.Module):
    def __init__(self):
        super(TextRNN_Att,self).__init__()
        self.embedding_dim = 100
        self.hidden_size = 50 #lstm的长度，可以和seq_legth一样，也可以比它长
        self.layer_num = 2
        self.class_num = 8
        self.attention_size = 256

        self.lstm = nn.LSTM(self.embedding_dim, # x的特征维度,即embedding_dim
                            self.hidden_size, #lstm的时间长度，这里可以表示为文本长度
                            self.layer_num, #把lstm作为一个整体，然后堆叠的个数的含义
                            batch_first=True,
                            bidirectional=True
                            )

        self.classificer = nn.Linear(self.hidden_size * self.layer_num, self.class_num)

    def attention(self,lstm_output):#lstm_output[batch_size, seq_length, hidden_size * direction_num]
        """
        :param lstm_output:
        :return: output
        这个是普通注意力机制attention的一种公式:
        M = tanh(H)
        a = softmax(WM)
        att_score = H*a
        上面的是矩阵形式
        """
        #初始化一个权重参数w_omega[hidden_size*layer_num,attention_size]
        #u_omega[attention_size,1]
        w_omega = nn.Parameter(torch.zeros(self.hidden_size*self.layer_num,self.attention_size)).to('cuda')
        u_omega = nn.Parameter(torch.zeros(self.attention_size,1)).to('cuda')

        #att_u[b,seq_length,attention_size]
        att_u = torch.tanh(torch.matmul(lstm_output,w_omega))


        # print('att_u',att_u)
        # print('att_u', att_u.size())

        #att_a[b, seq_length, 1]
        att_a = torch.matmul(att_u,u_omega)
        # print('att_a', att_a)
        # print('att_a', att_a.size())



        # att_score[b, seq_length, 1]
        att_score = F.softmax(att_a,dim=1)
        # print('att_score', att_score)
        # print('att_score', att_score.size())

        # att_output[b, seq_length, hidden_size * direction_num]
        att_output = lstm_output*att_score
        # print('att_output', att_output)
        # print('att_output', att_output.size())

        # output[b, hidden_size * direction_num]
        output = torch.sum(att_output,dim=1)
        # print('output', output)
        # print('output', output.size())

        return output


    def forward(self,x):
        #x的维度为(batch_size, time_step, input_size=embedding_dim)

        # 隐层初始化
        # h0维度为(num_layers*direction_num, batch_size, hidden_size)
        # c0维度为(num_layers*direction_num, batch_size, hidden_size)
        h0 = torch.zeros(self.layer_num*2,x.size(0),self.hidden_size).to('cuda') #定义一定要用torch.zeros(),torch.Tensor()只是定义了一个类型，并没有赋值
        c0 = torch.zeros(self.layer_num*2,x.size(0),self.hidden_size).to('cuda')

        #out维度为(batch_size, seq_length, hidden_size * direction_num)
        lstm_out,(hn,cn)  =self.lstm(x,(h0,c0))

        # attn_output[b, hidden_size * direction_num]
        attn_output = self.attention(lstm_out)#注意力机制


        logit = self.classificer(attn_output)


        return logit




