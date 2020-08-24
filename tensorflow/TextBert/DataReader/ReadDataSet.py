from transformers import BertTokenizer
from tqdm import tqdm
import os
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

class ReadDataSet():
    def __init__(self,data_file_name,args,repeat=1):
        self.max_sentence_length = args.max_sentence_length
        self.repeat = repeat
        self.tokenizer = BertTokenizer.from_pretrained(args.model_path)
        self.file_path = args.data_file_path
        self.file_name = data_file_name
        self.tf_tensors_list = self.readfiles_tokening()
        if 'train' in self.file_name:
            self.len_train = len(self.tf_tensors_list)


    def readfiles_tokening(self):
        file_path = os.path.join(self.file_path, self.file_name)
        tf_tensors_list = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines, desc='read data tokening'):
            line = line.strip().split('\t')
            input_ids,input_mask,label = self.do_process_data((line[0], line[1]))
            tf_tensors_list.append((input_ids,input_mask,label))
        return tf_tensors_list


    def __call__(self, *args, **kwargs):
        for (input_ids,input_mask,label) in self.tf_tensors_list:
            yield input_ids,input_mask,label

    def do_process_data(self, params):
        sentence = params[0]
        label = params[1]
        input_ids, input_mask = self.convert_into_indextokens_and_segment_id(sentence)
        input_ids = tf.constant(input_ids)
        input_mask = tf.constant(input_mask)
        label = tf.constant(label)
        return input_ids,input_mask,label

    def convert_into_indextokens_and_segment_id(self, text):
        tokeniz_text = self.tokenizer.tokenize(text[0:self.max_sentence_length])
        input_ids = self.tokenizer.convert_tokens_to_ids(tokeniz_text)
        input_mask = [1] * len(input_ids)

        pad_indextokens = [0] * (self.max_sentence_length - len(input_ids))
        input_ids.extend(pad_indextokens)
        input_mask_pad = [0] * (self.max_sentence_length - len(input_mask))
        input_mask.extend(input_mask_pad)
        return input_ids, input_mask

