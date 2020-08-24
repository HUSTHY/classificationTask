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
            # print('step',step)
            if step%100 ==0 and step%((int(train_len/args.batch_size/2/100))*100)!=0:
                logs = 'Epoch={},step={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{},best_valid_acc:{}'
                tf.print(tf.strings.format(logs, (
                    epoch, step, train_loss.result(), train_metric.result(), valid_loss.result(), valid_metric.result(),
                    best_valid_acc)))
                tf.print("")

            if step%((int(train_len/args.batch_size/2/100))*100)==0:
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
    print('*'*80,train_len)
    train_data = tf.data.Dataset.from_generator(train_generator,(tf.int32,tf.int32,tf.int32)).shuffle(buffer_size=1000).batch(args.batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    dev_generator = ReadDataSet('dev.tsv', args)
    dev_data = tf.data.Dataset.from_generator(dev_generator, (tf.int32, tf.int32, tf.int32)).shuffle(buffer_size=1000).batch(args.batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    model = TextBert(args)

    train_model(model,train_data,train_len,dev_data,args)






