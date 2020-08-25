from tensorflow.keras import metrics,optimizers,losses
import tensorflow as tf
from DataReader.ReadDataSet import ReadDataSet
from model.TextCNN import TextCNN
from tensorboardX import SummaryWriter
import argparse

"""
采用的是200个词语
loss可以下降的很快，但是验证集的准确率只能到65%

50个词语的可以试试(对应的就要修改模型结构)

z注意训练过程中还是要监控验证集的准确率——这个比较好！     

"""
writer = SummaryWriter('runs/exp')

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
        logical_gpus = tf.config.experimental.list_physical_devices('GPU')
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
def train_model(model,train_data,train_len,dev_data,args):
    optimizer = optimizers.Adam(learning_rate=args.lr)

    train_loss = metrics.Mean(name='train_loss')
    train_accuracy = metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

    valid_loss = metrics.Mean(name = 'valid_loss')
    valid_accuracy = metrics.SparseCategoricalAccuracy(name = 'valid_accuracy')

    loss_fun = losses.SparseCategoricalCrossentropy()
    step = 0
    for epoch in tf.range(args.epochs):
        for features,lables in train_data:
            train_step(model, features, lables, optimizer, train_loss, train_accuracy, loss_fun)
            step += 1
            # print('step',step)
            if step % 100 == 0 and step % ((int(train_len / args.batch_size / 2 / 100)) * 100) != 0:
                logs = 'Epoch={},step={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{},best_valid_acc:{}'
                tf.print(tf.strings.format(logs, (
                    epoch, step, train_loss.result(), train_accuracy.result(), valid_loss.result(), valid_accuracy.result(),
                    best_valid_acc)))

            if step % ((int(train_len / args.batch_size / 2 / 100)) * 100) == 0:
                for features, lables in dev_data:
                    valid_step(model, features, lables, optimizer, valid_loss, valid_accuracy, loss_fun)
                if valid_accuracy.result() >= best_valid_acc:
                    best_valid_acc = valid_accuracy.result()
                    save_path = args.model_save_path
                    # model.save(save_path,save_format='h5')
                    # model.save(save_path, save_format='tf')
                    model.save_weights(save_path, save_format='tf')
                logs = 'Epoch={},step={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{},best_valid_acc:{}'
                printbar()
                tf.print(tf.strings.format(logs, (
                    epoch, step, train_loss.result(), train_accuracy.result(), valid_loss.result(), valid_accuracy.result(),
                    best_valid_acc)))
                tf.print("")

        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()


@tf.function
def train_step(model, features, lables, optimizer, train_loss, train_accuracy, loss_fun):
    with tf.GradientTape() as tape:
        predicts = model(features)
        loss = loss_fun(lables,predicts)
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    train_loss.update_state(loss)
    train_accuracy.update_state(lables,predicts)


def valid_step(model, features, lables, optimizer, valid_loss, valid_accuracy, loss_fun):
    predicts = model(features)
    loss = loss_fun(lables,predicts)

    valid_loss.update_state(loss)
    valid_accuracy.update_state(lables,predicts)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='init params configuration')
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()
    print(args)
    train_generator = ReadDataSet('data/train_padding.csv')
    train_len = train_generator.len_train
    train_data = tf.data.Dataset.from_generator(train_generator,(tf.float32,tf.int32)).shuffle(buffer_size=2000).batch(batch_size=args.batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    dev_generator = ReadDataSet('data/dev_padding.csv')
    dev_data = tf.data.Dataset.from_generator(dev_generator, (tf.float32, tf.int32)).shuffle(
        buffer_size=2000).batch(batch_size=args.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model = TextCNN()

    train_model(model,train_data,train_len,dev_data,args)