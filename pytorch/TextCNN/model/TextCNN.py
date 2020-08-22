import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN,self).__init__()

        class_num = 8
        embedding_dim = 100
        ci = 1
        kernel_num = 25
        # kernel_sizes = [3,4,5]
        # self.convs = nn.ModuleList([nn.Conv2d(ci,kernel_num,(k,embedding_dim/2))for k in kernel_sizes])
        # #含义说明：nn.Conv2d(ci,kernel_num,(k,embedding_dim))
        # #ci就是输入的通道数目，是要和数据对的上的；kernel_num这里的意思就是输出通道数目；(k,embedding_dim)卷积核的形状，也就是2维度的k*embedding_dim
        # #nn.Conv2d(ci,cj,k)这里的K就是表示卷积核的形状是正方形的，k*k

        self.conv1 = nn.Conv2d(ci, kernel_num, (3, int(embedding_dim))) #这里一定要输入4维向量[B,C,L,D]
        self.conv2 = nn.Conv2d(ci, kernel_num, (5, int(embedding_dim)))
        self.conv3 = nn.Conv2d(ci, kernel_num, (7, int(embedding_dim)))
        self.conv4 = nn.Conv2d(ci, kernel_num, (9, int(embedding_dim)))

        self.dropout = nn.Dropout(0.5)#丢掉10%
        self.classificer = nn.Linear(kernel_num*4,class_num)

    def conv_and_pool(self, x, conv):
        #(B, Ci, L, D)
        x = F.relu(conv(x))#(B,kernel_num,L-3+1,D-D+1)
        x = x.squeeze(3)# (B, kernel_num, L-3+1)
        x = F.max_pool1d(x, x.size(2))#(B, kernel_num,1)
        x = x.squeeze(2)# (B,kernel_num) squeeze压缩维度
        return x

    def forward(self,x):
        #size(B,L,D)
        x = x.unsqueeze(1)  #(B, Ci, L, D)#unsqueeze增加维度

        x1 = self.conv_and_pool(x, self.conv1)  # (B,kernel_num)
        x2 = self.conv_and_pool(x, self.conv2)  # (B,kernel_num)
        x3 = self.conv_and_pool(x, self.conv3)  # (B,kernel_num)
        x4 = self.conv_and_pool(x, self.conv4)  # (B,kernel_num)

        x = torch.cat((x1, x2, x3,x4), 1)  # (B,len(Ks)*kernel_num)
        x = self.dropout(x)  # (B, len(Ks)*kernel_num)
        logit = self.classificer(x)  # (B, C)
        return logit




