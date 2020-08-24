import tensorflow as tf
from tensorflow.keras import optimizers,metrics,losses
from tqdm import tqdm
from model.TextBert import TextBert
import argparse
from DataReader.ReadDataSet import ReadDataSet

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # 设置使用哪一块GPU（默认是从0开始）

# 下面就是实现按需分配的代码！
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

"""
训练过程中还是要监控验证集准确率
"""
# writer = SummaryWriter('runs/exp')
# def train(model,train_iter,dev_iter,args):
#     model.to('cuda')
#
#     if args.requires_grad:
#         param_optimizer = list(model.named_parameters())
#         no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#         # 设置模型参数的权重衰减
#         optimizer_grouped_parameters = [
#             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#              'weight_decay': 0.01},
#             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#         ]
#         # t_total = len(train_iter)
#         # # 学习率的设置
#         # optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
#         # scheduler = get_linear_schedule_with_warmup(
#         #     optimizer, num_warmup_steps=100, num_training_steps=t_total
#         # )
#         # AdamW 这个优化器是主流优化器
#         optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)
#         scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, min_lr=1e-7, patience=5, verbose=True,
#                                       eps=1e-8)  # mode max表示当监控量停止上升时，学习率将减小；min表示当监控量停止下降时，学习率将减小；这里监控的是dev_acc因此应该用max
#
#     else:
#         # 初始学习率
#         optimizer_params = {'lr': 1e-3, 'eps': 1e-8}
#         optimizer = AdamW(model.parameters(), **optimizer_params)
#         scheduler = ReduceLROnPmax_sentence_lengthlateau(optimizer, mode='max', factor=0.8, min_lr=1e-6, patience=2, verbose=True,
#                                       eps=1e-8)  # mode max表示当监控量停止上升时，学习率将减小；min表示当监控量停止下降时，学习率将减小；这里监控的是dev_acc因此应该用max
#
#     early_stop_step = 20000
#     epochs = 200
#     last_improve = 0 #记录上次提升的step
#     flag = False  # 记录是否很久没有效果提升
#     dev_best_acc = 0
#     dev_loss = float(50)
#     dev_acc = 0
#     correct = 0
#     total = 0
#     global_step = 0
#     for epoch in range(epochs):
#         for step,batch in enumerate(tqdm(train_iter,desc='Train iteration:')):
#             global_step += 1
#             optimizer.zero_grad()
#             batch = tuple(t.to('cuda') for t in batch)
#             input_ids = batch[0]
#             input_mask = batch[1]
#             label = batch[2]
#             model.train()
#             output = model(input_ids,input_mask)
#             loss = F.cross_entropy(output,label)
#             loss.backward()
#             optimizer.step()
#             total += label.size(0)
#             _,predict = torch.max(output,1)
#             correct += (predict==label).sum().item()
#             train_acc = correct / total
#             if (step+1)%10 == 0:
#                 print('Train Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,loss:{:.6f}'.format(epoch,epochs,step,len(train_iter),train_acc*100,loss.item()))
#             if (step+1)%(len(train_iter)/5)==0:#(step不能+1，dev_acc和dev_loss的初始化)
#                 dev_acc,dev_loss = dev(model, dev_iter)
#                 dev_loss = dev_loss.item()
#                 if dev_best_acc < dev_acc:
#                     dev_best_acc = dev_acc
#                     path = 'savedmodel/TextBert_model.pkl'
#                     torch.save(model,path)
#                     last_improve = global_step
#                 print("DEV Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,dev_acc{:.6f} %,best_dev_acc{:.6f} %,train_loss:{:.6f},dev_loss:{:.6f}".format(epoch, epochs, step, len(train_loader), train_acc * 100, dev_acc * 100,dev_best_acc*100,loss.item(),dev_loss))
#             if global_step-last_improve >= early_stop_step:
#                 print("No optimization for a long time, auto-stopping...")
#                 flag = True
#                 break
#             writer.add_scalar('train_loss', loss.item(), global_step=global_step)
#             writer.add_scalar('dev_loss', dev_loss, global_step=global_step)
#             writer.add_scalar('train_acc', train_acc, global_step=global_step)
#             writer.add_scalar('dev_acc', dev_acc, global_step=global_step)
#         scheduler.step(dev_best_acc)
#         if flag:
#             break
#     writer.close()
#
# def dev(model, dev_iter):
#     model.eval()
#     loss_total = 0
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for step,batch in enumerate(tqdm(dev_iter,desc='dev iteration:')):
#             batch = tuple(t.to('cuda') for t in batch)
#             input_ids = batch[0]
#             input_mask = batch[1]
#             label = batch[2]
#             output = model(input_ids,input_mask)
#             loss = F.cross_entropy(output, label)
#             loss_total += loss
#             total += label.size(0)
#             _, predict = torch.max(output, 1)
#             correct += (predict == label).sum().item()
#         res = correct/total
#         return res,loss_total/len(dev_iter)


def printbar():
    today = tf.timestamp()%(24*60*60)

    hour = tf.cast(today // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today % 60), tf.int32)
    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m))==1:
            return (tf.strings.format("0{}", m))
        else:
            return (tf.strings.format("{}", m))

    timestrins = tf.strings.join([timeformat(hour), timeformat(minite),
                                  timeformat(second)], separator=":")
    tf.print('====' * 20 + timestrins)
@tf.function
def train_step(model,input_ids,input_mask,labels,optimizer,train_loss,train_metric,loss_fun):
    with tf.GradientTape() as tape:
        predictions = model({'input_ids':input_ids,'attention_mask':input_mask})
        loss = loss_fun(labels,predictions)
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)

def valid_step(model, input_ids, input_mask, labels, optimizer, valid_loss, valid_metric,loss_fun):
    predictions = model({'input_ids':input_ids,'attention_mask':input_mask})
    batch_loss = loss_fun(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)

def train_model(model,train_data,train_len,dev_data,args):
    optimizer = optimizers.Adam(learning_rate=args.lr)
    train_loss = metrics.Mean(name='train_loss')
    train_metric = metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = metrics.Mean(name='valid_loss')
    valid_metric = metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    loss_fun = losses.SparseCategoricalCrossentropy()
    step = 0
    best_valid_acc = 0
    for epoch in tf.range(args.epochs):
        for input_ids,input_mask,labels in train_data:
            train_step(model,input_ids,input_mask,labels,optimizer,train_loss,train_metric,loss_fun)
            step += 1
            print('step',step)
            if step%(train_len/args.batch_size/2)==0:
                for input_ids, input_mask, labels in dev_data:
                    valid_step(model, input_ids, input_mask, labels, optimizer, valid_loss, valid_metric,loss_fun)
                if valid_metric.result()>=best_valid_acc:
                    best_valid_acc = valid_metric.result()
                    save_path = args.model_save_path
                    # model.save(save_path,save_format='h5')
                    # model.save(save_path, save_format='tf')
                    model.save_weights(save_path,save_format='tf')
                logs = 'Epoch={},step={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{},best_valid_acc:{}'
                printbar()
                tf.print(tf.strings.format(logs, (
                epoch,step,train_loss.result(), train_metric.result(), valid_loss.result(), valid_metric.result(),best_valid_acc)))
                tf.print("")






        train_loss.reset_states()
        train_metric.reset_states()
        valid_loss.reset_states()
        valid_metric.reset_states()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='init params configuration')
    parser.add_argument('--batch_size',type=int,default=100)
    parser.add_argument('--model_path',type=str,default='./pretrain_model/')
    parser.add_argument('--requires_grad', type= bool,default=True)
    parser.add_argument('--data_file_path',type=str,default='data_set/patent')
    parser.add_argument('--max_sentence_length',type=int,default=400)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--model_save_path', type=str, default='savedmodel/')
    args = parser.parse_args()
    print(args)

    train_generator = ReadDataSet('train.tsv',args)
    train_len = train_generator.len_train
    print('len(train_len)',train_len)
    train_data = tf.data.Dataset.from_generator(train_generator,(tf.int32,tf.int32,tf.int32)).shuffle(buffer_size=1000).batch(args.batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    dev_generator = ReadDataSet('dev.tsv', args)
    dev_data = tf.data.Dataset.from_generator(dev_generator, (tf.int32, tf.int32, tf.int32)).shuffle(buffer_size=1000).batch(args.batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    model = TextBert(args)

    train_model(model,train_data,train_len,dev_data,args)






