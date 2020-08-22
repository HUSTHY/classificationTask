# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from data_process import getdatafromjson_to_csv



import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset


from transformers_local import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from transformers_local import glue_convert_examples_to_features as convert_examples_to_features

from transformers_local import InputExample


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter





path = './output/'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'





def model_init():
    model.to(device)
    return model


def _create_examples(lines):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % ('predict', i)
        if len(line)>1:
            text_a = line
            label = 0
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label= label))
    return examples


def produce_dataloder(text):
    examples = _create_examples(text)
    label_list = [0]
    features = convert_examples_to_features(
        examples, tokenizer, max_length=128, label_list=label_list, output_mode='classification',
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    predict_sampler = SequentialSampler(dataset)
    predict_dataloader = DataLoader(dataset, sampler=predict_sampler, batch_size=48)

    return predict_dataloader




def predict(model,dataloader):
    preds = None
    for batch in tqdm(dataloader,desc="Predicting"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            outputs = model(**inputs)
            logits = outputs[0]
        if preds is None:
            preds = logits.cpu().numpy()
        else:
            preds = np.append(preds, logits.cpu().numpy(), axis=0)
    preds = np.argmax(preds, axis=1)
    return preds


def do_predict(texts):
    model = model_init()
    dataloader = produce_dataloder(texts)
    result = predict(model,dataloader)
    return result

# if __name__ == "__main__":
    # text = ['一体化音响','监测预警系统','一全球首创','智能PDU设备(PB01A/PB02A)', '煎烤器在线测试系统','每条可录制时间不低于15秒,总时间最大可超过', '测试仪重量']
    # text1 = ['智能PDU设备(PB01A/PB02A)', '煎烤器在线测试系统,']
    # text2 = ['每条可录制时间不低于15秒,总时间最大可超过', '测试仪重量']
    # text3 = ['路面稽查企业巡查APP', '天宝yuma2']
    # # text_list = [text0,text1,text2,text3]
    # 'param:text,一个list'
    # model = model_init()
    # # for text in text_list:
    # print(type(text))
    # print(text)
    # print(len(text))
    # dataloader = produce_dataloder(text)
    # result = predict(model, dataloader)
    # print(result)
    # print(len(result))

    # model = model_init()
    # df = getdatafromjson_to_csv()[0:100]
    # text = list(df['text'])
    # print(type(text))
    # print(text)
    # print(len(text))
    # dataloader = produce_dataloder(text)
    # result = predict(model, dataloader)
    # print(result)
    # print(len(result))