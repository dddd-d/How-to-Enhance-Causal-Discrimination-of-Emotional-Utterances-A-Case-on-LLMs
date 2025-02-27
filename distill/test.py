from sklearn.metrics import f1_score
from transformers import T5ForConditionalGeneration,GPT2LMHeadModel
from transformers import AutoTokenizer
import numpy as np
import re
from tqdm import tqdm
MODEL = 'gpt2'
if MODEL == 't5':
    tokenizer = AutoTokenizer.from_pretrained('ckpts/t5-v1_1-base/0.5/1024/8/AdamW/5e-05/0',use_fast=False)
    model = T5ForConditionalGeneration.from_pretrained('ckpts/t5-v1_1-base/0.5/1024/8/AdamW/5e-05/0')
else:
    tokenizer = AutoTokenizer.from_pretrained('ckpts/gpt2/0.5/300/2/AdamW/5e-05/0/checkpoint-2000',use_fast=False)
    model = GPT2LMHeadModel.from_pretrained('ckpts/gpt2/0.5/300/2/AdamW/5e-05/0/checkpoint-2000')
    tokenizer.pad_token = tokenizer.eos_token

def generate(str):
    model_inputs = tokenizer(str,max_length=512,truncation=True, return_tensors='pt')
    outputs = model.generate(**model_inputs,output_hidden_states=False, max_new_tokens=512, pad_token_id=model.config.eos_token_id)
    if MODEL == 't5':
        predictions = np.where(outputs[0] != -100, outputs[0], tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    else:
        decoded_preds = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
    return decoded_preds

def extract(str):
    if MODEL == 't5':
        match = re.search(r"\] : (.*?) 2\.", str)
    else:
        match = re.search(r"\]: (.*?)\n2\.", str)
    if match:
        e = match.group(1).strip()
        print(e)  # 输出 "John and the listen er had  a great vacation ."
    else:
        e = ''
        print("未找到匹配内容")
    return e

import json
test = []
with open('data/test-for-distill2.json','r') as f:
    for line in f:
        test.append(json.loads(line))

true_label = []
pre_label = []
res = []
for i in tqdm(range(len(test))):
    d = test[i]
    type = d['type']
    x1 = 'Answer the following questions directly:\n' + d['x']
    x2 = 'Think and answer the following questions step by step:\n' + d['x']
    y_c = d['y_c']
    y = d['y']
    if MODEL == 't5':
        y_p = generate(x1)
        y_e = ' '.join(generate(x2))
    else: 
        y_e = ''.join(generate(x2))
    e = extract(y_e)
    d['E_i'] = e
    if e != '':
        res.append(d)
    if type == 'pos':
        true_label.append([1,0])
    else:
        true_label.append([0,1])
    if MODEL == 't5':
        if 'Yes' in y_p:
            pre_label.append([1,0])
        else:
            pre_label.append([0,1])
    else:
        if 'the answer is yes' in y_e:
            pre_label.append([1,0])
        else:
            pre_label.append([0,1])
     
true_label = np.array(true_label)
pre_label = np.array(pre_label)

f1 = f1_score(true_label, pre_label, average=None)
print(f1[0], f1[1], f1.mean())

print(len(res))
with open('data/test-gpt2.json','w') as f:
    for r in res:
        json.dump(r, f)
        f.write('\n')



