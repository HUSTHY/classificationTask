from Code.ReadDataSet import ReadDataSet
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from transformers import BertModel
import argparse
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from lightgbm import LGBMClassifier

def get_bert_vector(model,train_iter,dev_iter,args):
    model.to('cuda')
    model.eval()

    train_vec = []
    train_label = []
    dev_vec = []
    dev_label = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_iter, desc='dev iteration:')):
            batch = tuple(t.to('cuda') for t in batch)
            input_ids = batch[0]
            input_mask = batch[1]
            label = batch[2]
            output = model(input_ids, input_mask)[1]#[1]pooler_output,就是cls对应的那个向量,[0]last_hidden_state，这个需要自己取处理才行
            label = label.to('cpu').numpy().tolist()
            output = output.to('cpu').numpy().tolist()
            dev_vec.extend(output)
            dev_label.extend(label)



        for step, batch in enumerate(tqdm(train_iter, desc='train iteration:')):
            batch = tuple(t.to('cuda') for t in batch)
            input_ids = batch[0]
            input_mask = batch[1]
            label = batch[2]
            output = model(input_ids, input_mask)[1]
            label = label.to('cpu').numpy().tolist()
            output = output.to('cpu').numpy().tolist()
            train_vec.extend(output)
            train_label.extend(label)
    return train_vec,train_label,dev_vec,dev_label




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='init params configuration')
    parser.add_argument('--batch_size',type=int,default=100)
    parser.add_argument('--model_path',type=str,default='./pretrain_model/Chinese-BERT-wwm')
    parser.add_argument('--requires_grad', type= bool,default=True)
    parser.add_argument('--data_file_path',type=str,default='Data/patent')
    parser.add_argument('--max_sentence_length',type=int,default=400)
    args = parser.parse_args()
    print(args)

    train_data = ReadDataSet('train.tsv',args)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

    dev_data = ReadDataSet('dev.tsv',args)
    dev_loader = DataLoader(dataset=dev_data, batch_size=args.batch_size, shuffle=True)

    bert_model = BertModel.from_pretrained(args.model_path)
    train_vec, train_label, dev_vec, dev_label = get_bert_vector(bert_model,train_loader,dev_loader,args)


    clf = LGBMClassifier()
    clf.fit(train_vec, train_label)
    dev_pred = clf.predict(dev_vec)
    f1 = f1_score(dev_label, dev_pred, average='macro')
    pre = precision_score(dev_label, dev_pred, average='micro')
    acc = accuracy_score(dev_label, dev_pred)
    recall = recall_score(dev_label, dev_pred, average='micro')
    print('LGBMClassifier f1:', f1)
    print('LGBMClassifier pre:', pre)
    print('LGBMClassifier recall:', recall)
    print('LGBMClassifier acc:', acc)




    clf = GaussianNB()
    clf.fit(train_vec, train_label)
    dev_pred = clf.predict(dev_vec)
    f1 = f1_score(dev_label, dev_pred, average='macro')
    pre = precision_score(dev_label, dev_pred, average='micro')
    acc = accuracy_score(dev_label, dev_pred)
    recall = recall_score(dev_label, dev_pred, average='micro')
    print('GaussianNB f1:', f1)
    print('GaussianNB pre:', pre)
    print('GaussianNB recall:', recall)
    print('GaussianNB acc:', acc)