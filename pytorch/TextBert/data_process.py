import  json
import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# def isChineseContain(text):
#     CHmodel = re.compile(u'[\u4e00-\u9fa5]')
#     match = CHmodel.search(text)
#     if match:
#         return text
#     else:
#         return None
#
#
# def getdatafromjson_to_csv():
#     path = 'data_set/crawling_noisy_product'
#     listidr = os.listdir(path)
#     company_names = []
#     texts = []
#
#     for ele in listidr:
#         file_folder_path = os.path.join(path,ele)
#         files_list = os.listdir(file_folder_path)
#         for file in files_list:
#             company_name = file.split('_')[1]
#             file_path = os.path.join(file_folder_path,file)
#             with open(file_path,'r') as f:
#                 reader = json.load(f)
#             for dic in reader:
#                 text = dic.get('word','无')
#                 company_names.append(company_name)
#                 texts.append(text)
#
#     df = pd.DataFrame()
#     df['company_name'] = company_names
#     df['text'] = texts
#     df.drop_duplicates(inplace=True)
#     df.to_csv('result/data.csv',index=False)
#     return df

if __name__ == '__main__':
    with open('data_set/发明专利数据.json', 'r') as f:
        patents = json.load(f,strict=False)
    # vacabulary = ['A','B','C']
    pat_summarys = []
    ipc_classes = []
    text_length = []
    ipc_class_first_tokens = []
    for ele in patents:
        pat_summary = ele['pat_summary'].replace('\t','').replace('\n','')
        ipc_class = ele['ipc_class']
        ipc_class_first_token = ipc_class[0]
        if not ipc_class_first_token.isdigit():
            pat_summarys.append(pat_summary)
            ipc_classes.append(ipc_class)
            ipc_class_first_tokens.append(ipc_class_first_token)
            text_length.append(len(pat_summary))
    ipc_class_first_tokens_set = sorted(list(set(ipc_class_first_tokens)))
    print(ipc_class_first_tokens[0:10])
    print(ipc_class_first_tokens_set)
    label = []
    for ele in ipc_class_first_tokens:
        label.append(ipc_class_first_tokens_set.index(ele))

    #统计文本的长度
    text_length = sorted(text_length, reverse=True)
    count = 0 #长度大于510的数量
    for ele in text_length:
        if ele > 510:
            count += 1
        else:
            break
    print(count)
    #可视化，直方图
    plt.hist(text_length, bins=50, normed=1, facecolor="blue", edgecolor="black", alpha=0.7,cumulative=True)
    plt.xlabel("文本区间")
    # 显示纵轴标签
    plt.ylabel("频数/频率")
    plt.show()


    df = pd.DataFrame()
    df['text'] = pat_summarys
    df['ipc_class'] = ipc_classes
    df['first_IPC'] = ipc_class_first_tokens
    df['label'] = label
    print(len(df))
    df.dropna(inplace=True)
    print(len(df))
    df = df[df['text'].str.len()<=300][['text','label']]
    print(len(df))


    # df.to_csv('data_set/patent/all.csv',index=False)
    train_df,test_df = train_test_split(df,test_size=0.2)
    print(len(train_df))
    print(len(test_df))
    train_df.to_csv('data_set/patent/train.tsv',sep='\t',index=False,header=False)
    test_df.to_csv('data_set/patent/dev.tsv', sep='\t', index=False, header=False)

    print(set(label))




