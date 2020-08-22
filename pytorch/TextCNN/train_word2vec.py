import pandas as pd
from gensim.models import Word2Vec

#文本向量化的时候回出现voo的问题，提前把word2vec做增量训练，得到新的模型

if __name__ == '__main__':
    train_text = list(pd.read_csv('data/train_word_cut.csv')['text_word_cut'])
    dev_text = list(pd.read_csv('data/dev_word_cut.csv')['text_word_cut'])
    train_text.extend(dev_text)
    model = Word2Vec.load('pretrain_model/word2vec/word2vec_newtrain.model')
    #增量训练Word2Vec，只训练新词，不训练旧词
    model.build_vocab(train_text,update=True)
    model.train(train_text, total_examples=model.corpus_count, epochs=5)
    model.save('pretrain_model/word2vec/word2vec_newtrain.model')
