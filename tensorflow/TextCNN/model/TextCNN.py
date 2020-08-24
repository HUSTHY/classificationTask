import tensorflow as tf
from tensorflow.keras import layers,models
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class TextCNN(models.Model):
    def __init__(self):
        super(TextCNN,self).__init__()

        class_num = 8
        embedding_dim = 100
        ci = 1
        kernel_num = 25

        self.conv1 = layers.Conv2D()
        self.conv2 = layers.Conv2D()
        self.conv3 = layers.Conv2D()
        self.conv4 = layers.Conv2D()

        self.dropout = layers.Dropout(0.8)
        self.classificer = layers.Dense(class_num,activation='softmax')

    def conv_and_pool(self, x, conv):
        #(B, Ci, L, D)
        x = tf.nn.relu(conv(x))#(B,kernel_num,L-3+1,D-D+1)
        x = x.squeeze(3)# (B, kernel_num, L-3+1)
        x = tf.nn.max_pool1d(x, x.size(2))#(B, kernel_num,1)
        x = x.squeeze(2)# (B,kernel_num) squeeze压缩维度
        return x

    def forward(self,x):
        #size(B,L,D)
        x = x.unsqueeze(1)  #(B, Ci, L, D)#unsqueeze增加维度

        x1 = self.conv_and_pool(x, self.conv1)  # (B,kernel_num)
        x2 = self.conv_and_pool(x, self.conv2)  # (B,kernel_num)
        x3 = self.conv_and_pool(x, self.conv3)  # (B,kernel_num)
        x4 = self.conv_and_pool(x, self.conv4)  # (B,kernel_num)

        x = tf.cast((x1, x2, x3,x4), 1)  # (B,len(Ks)*kernel_num)
        x = self.dropout(x)  # (B, len(Ks)*kernel_num)
        logit = self.classificer(x)  # (B, C)
        return logit




