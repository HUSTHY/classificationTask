from data_process import getdatafromjson_to_csv
from predict_classification import do_predict


if __name__ == '__main__':
    df = getdatafromjson_to_csv()
    texts = list(df['text'])
    result = do_predict(texts)
    print(df)
    df['predict_label'] = result
    df = df[df['predict_label']==1]
    print(len(df))
    df.to_csv('result/re.csv',index=False)
    print('finshed!')