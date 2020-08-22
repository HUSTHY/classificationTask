from DataReader.ReadDataSet import ReadDataSet
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
from tqdm import tqdm
import torch.nn.functional as F
from model.TextCNN import TextCNN
from tensorboardX import SummaryWriter
"""
采用的是200个词语
loss可以下降的很快，但是验证集的准确率只能到65%

50个词语的可以试试(对应的就要修改模型结构)

z注意训练过程中还是要监控验证集的准确率——这个比较好！

"""
writer = SummaryWriter('runs/exp')

def train(model,train_iter,dev_iter):
    model.to('cuda')
    #初始学习率
    optimizer_params = {'lr': 1e-3, 'eps': 1e-8}
    optimizer = torch.optim.Adam(model.parameters(),**optimizer_params)
    scheduler = ReduceLROnPlateau(optimizer,mode='max',factor=0.8,min_lr=1e-5,patience=5,verbose=True,eps=1e-8)#mode max表示当监控量停止上升时，学习率将减小；min表示当监控量停止下降时，学习率将减小；这里监控的是loss因此应该用min
    early_stop_step = 1000
    epochs = 200
    last_improve = 0 #记录上次提升的step
    flag = False  # 记录是否很久没有效果提升
    dev_best_loss = float('inf')
    dev_best_acc = 0
    correct = 0
    total = 0
    global_step = 0
    for epoch in range(epochs):
        for step,batch in tqdm(enumerate(train_iter),desc='Train iteration:'):
            global_step += 1
            optimizer.zero_grad()
            batch = tuple(t.to('cuda') for t in batch)
            input = batch[0]
            label = batch[1]
            model.train()
            output = model(input)
            loss = F.cross_entropy(output,label)
            loss.backward()
            optimizer.step()

            total += label.size(0)
            _,predict = torch.max(output,1)
            correct += (predict==label).sum().item()

            train_acc = correct / total
            if (step+1)%20 == 0:
                print('Train Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,loss:{:.6f}'.format(epoch,epochs,step,len(train_iter),train_acc*100,loss.item()))
            if  (step)%(len(train_iter)/2)==0:
                dev_acc,dev_loss = dev(model, dev_iter)
                if dev_best_acc < dev_acc:
                    dev_best_acc = dev_acc
                    path = 'savedmodel/TextCNN_model.pkl'
                    torch.save(model,path)
                    last_improve = global_step
                print("DEV Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,dev_acc{:.6f} %,best_dev_acc{:.6f} %,train_loss:{:.6f},dev_loss:{:.6f}".format(epoch, epochs, step, len(train_loader), train_acc * 100, dev_acc * 100,dev_best_acc * 100, loss.item(),dev_loss.item()))

            if global_step-last_improve >= early_stop_step:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            writer.add_scalar('train_loss', loss.item(), global_step=global_step)
            writer.add_scalar('dev_loss', dev_loss.item(), global_step=global_step)
            writer.add_scalar('train acc', train_acc, global_step=global_step)
            writer.add_scalar('dev acc', dev_acc, global_step=global_step)
        scheduler.step(dev_best_acc)
        if flag:
            break
    writer.close()

def dev(model, dev_iter):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for step,batch in tqdm(enumerate(dev_iter),desc='dev iteration:'):
            batch = tuple(t.to('cuda') for t in batch)
            input = batch[0]
            label = batch[1]
            output = model(input)
            loss = F.cross_entropy(output, label)
            loss_total += loss
            total += label.size(0)
            _, predict = torch.max(output, 1)
            correct += (predict == label).sum().item()
        res = correct/total
        return res,loss_total/len(dev_iter)


if __name__ == '__main__':
    batch_size = 2000
    train_data = ReadDataSet('data/train_padding.csv')
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    dev_data = ReadDataSet('data/dev_padding.csv')
    dev_loader = DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=True)
    model = TextCNN()

    train(model,train_loader,dev_loader)