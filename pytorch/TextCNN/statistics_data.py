import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    train_text = list(pd.read_csv('data/train_word_cut.csv')['text_word_cut'])
    dev_text = list(pd.read_csv('data/dev_word_cut.csv')['text_word_cut'])
    all_text_length = []
    for ele in train_text:
        ele = ele.split(' ')
        all_text_length.append(len(ele))

    for ele in dev_text:
        ele = ele.split(' ')
        all_text_length.append(len(ele))

    plt.hist(all_text_length, 1000, normed=1, cumulative=True)
    plt.show()


"""
统计得出文本词数目在50-200之间，可以把200词之外的数据截取掉，不足200 词的全部补充特殊字符0

"""
