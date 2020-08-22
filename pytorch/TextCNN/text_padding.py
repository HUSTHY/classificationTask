import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm
from multiprocessing import Pool
"""
分词后的文本做padding操作！方便后续直接形成word2vec向量操作，顺便也
把word2vec外面的词语给去除掉————发现时间还是要40个小时，得用多进程了。
"""


def function(params):
    text = params[0]
    wv_words = params[1]
    keep_words = []
    words = text.split(' ')
    for word in words:
        if word in wv_words:
            keep_words.append(word)
        if len(keep_words) >= 200:
            break
    if len(keep_words) < 200:
        padding = ['0'] * (200 - len(keep_words))
        keep_words.extend(padding)
    content = ' '.join(keep_words)
    return content



def text_padding():
    wv_model  = Word2Vec.load('pretrain_model/word2vec/word2vec_newtrain.model')
    wv_words = wv_model.wv.index2word
    train = pd.read_csv('data/train_word_cut.csv')
    train_text = list(train['text_word_cut'])

    train_params = []
    for text in train_text:
        train_params.append((text,wv_words))

    with Pool(12) as pool:
        new_train_text = list(tqdm(pool.imap(function,train_params),total=len(train_params), desc='train set padding:'))
    pool.close()
    pool.join()

    train['text_padding'] = new_train_text
    train = train[['text_padding', 'label']]
    train.to_csv('data/train_padding.csv', index=False)

    dev = pd.read_csv('data/dev_word_cut.csv')
    dev_text = list(dev['text_word_cut'])

    dev_params = []
    for text in dev_text:
        dev_params.append((text, wv_words))

    with Pool(12) as pool:
        new_dev_text = list(tqdm(pool.imap(function, dev_params), total=len(dev_params), desc='dev set padding:'))
    pool.close()
    pool.join()

    dev['text_padding'] = new_dev_text
    dev = dev[['text_padding', 'label']]
    dev.to_csv('data/dev_padding.csv', index=False)



if __name__ == '__main__':
    text_padding()