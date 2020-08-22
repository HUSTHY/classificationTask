from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
from tqdm import tqdm
import os
import logging
logger = logging.getLogger(__name__)

class ReadDataSet(Dataset):
    def __init__(self,data_file_name,args,repeat=1):
        self.max_sentence_length = args.max_sentence_length
        self.repeat = repeat
        self.tokenizer = BertTokenizer.from_pretrained(args.model_path)
        self.process_data_list = self.read_file(args.data_file_path,data_file_name)


    def read_file(self,file_path,file_name):
        file_name_sub = file_name.split('.')[0]
        file_cach_path = os.path.join(file_path,"cached_{}".format(file_name_sub))
        if os.path.exists(file_cach_path):#直接从cach中加载
            logger.info('Load tokenizering from cached file %s', file_cach_path)
            process_data_list = torch.load(file_cach_path)
            return process_data_list
        else:
            file_path = os.path.join(file_path,file_name)
            data_list = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
            for line in tqdm(lines, desc='read data'):
                line = line.strip().split('\t')
                data_list.append((line[0], line[1]))
            process_data_list = []
            for ele in tqdm(data_list, desc="Tokenizering"):
                res = self.do_process_data(ele)
                process_data_list.append(res)
            logger.info('Saving tokenizering into cached file %s',file_cach_path)
            torch.save(process_data_list,file_cach_path)#保存在cach中
            return process_data_list


    def do_process_data(self, params):

        res = []
        sentence = params[0]
        label = params[1]

        input_ids, input_mask = self.convert_into_indextokens_and_segment_id(sentence)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)

        label = torch.tensor(int(label))

        res.append(input_ids)
        res.append(input_mask)
        res.append(label)

        return res

    def convert_into_indextokens_and_segment_id(self, text):
        tokeniz_text = self.tokenizer.tokenize(text[0:self.max_sentence_length])
        input_ids = self.tokenizer.convert_tokens_to_ids(tokeniz_text)
        input_mask = [1] * len(input_ids)

        pad_indextokens = [0] * (self.max_sentence_length - len(input_ids))
        input_ids.extend(pad_indextokens)
        input_mask_pad = [0] * (self.max_sentence_length - len(input_mask))
        input_mask.extend(input_mask_pad)
        return input_ids, input_mask

    def __getitem__(self, item):
        input_ids = self.process_data_list[item][0]
        input_mask = self.process_data_list[item][1]
        label = self.process_data_list[item][2]
        return input_ids, input_mask,label

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.process_data_list)
        return data_len


