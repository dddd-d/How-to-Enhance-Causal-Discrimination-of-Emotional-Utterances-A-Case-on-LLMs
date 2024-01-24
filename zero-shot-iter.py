import os
import re
import numpy as np
import torch
from openai import OpenAI
import json
from sentence_transformers import SentenceTransformer,util
import random
import pickle as pkl
import json

MODEL = "gpt-3.5-turbo" #gpt-3.5-turbo  gpt-4-1106-preview
SEED = 123
MAX_TOKENS = 1024
EPOCHS = 5
NUM_PAIRS = 200

ALL_result = []
all_result = []

PROMPT = 'zero-shot-iter'  

from datetime import datetime
current_datetime = datetime.now()
current_date = current_datetime.date()
dir1 = PROMPT + "_result_"+'epoch'+str(EPOCHS)+'_pair'+str(NUM_PAIRS)+'_'+str(current_date)
os.makedirs(dir1, exist_ok=True)

error = 0
error_type = {'存在性':0,'方向性':0}
error_number = 0

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
client = OpenAI()
client.api_key = os.environ.get("OPENAI_API_KEY")
print("OpenAI API Key:", os.environ.get("OPENAI_API_KEY"))

with open('dailydialog_train.json','r') as f:
  data = json.load(f)
#data[0]={'A': 'Guess who I saw yesterday ?', 'B': 'Avril Lavigen !', 'A_emotion': 'neutral', 'B_emotion': 'happiness'}

for i in range(len(data)): #len(data)
    y1 = error
    y2 = error_type
    try:
        num = i
        result = {} 
        judge = {} 
        
        print('\n================NO.{} sample================\n'.format(str(i+1)))
        pair = data[num]
        A = pair['A']
        B = pair['B']
        A_emotion = pair['A_emotion']
        B_emotion = pair['B_emotion']
        true_label = pair['label']

        result['A'] = A
        result['B'] = B
        result['true_label'] = true_label
        judge['true_label'] = true_label

        messages_no_implicit = []
        prompt_no_implicit = "There are two utterances: (1) " + A + "(2) " + B + "\
                              Is there a emotional cause-and-effect relationship between utterance (1) and utterance (2)? Answer Yes or No directly."
        messages_no_implicit.append({"role": "user", "content": prompt_no_implicit})

        result['Q1'] = prompt_no_implicit

        completion_no_implicit = client.chat.completions.create(
            model=MODEL,
            messages=messages_no_implicit,
            max_tokens=MAX_TOKENS,
            seed=SEED,
            )
        chat_response_no_implicit = completion_no_implicit
        answer_no_implicit = chat_response_no_implicit.choices[0].message.content
        result['A1'] = answer_no_implicit
        messages_no_implicit.append({"role": "assistant", "content": answer_no_implicit})
        print(f'ChatGPT: {answer_no_implicit}')

        prompt_iter = "Example:\
                       '''''' \
                       Here are two utterances: (1) But really, maybe this can work. Everyone loves coffee, and everyone needs motor oil.(2) A match made in heaven. \
                       Is there a emotional cause-and-effect relationship between utterance (1) and utterance (2)?  Let's think step by step.\
                       Answer: To determine if there is an emotional cause-and-effect relationship between utterance (1) and utterance (2), let's analyze the statements step by step:\
                       Step 1. Utterance (1): 'But really, maybe this can work. Everyone loves coffee, and everyone needs motor oil.'\
                       This statement suggests optimism or hope that a certain idea or plan could be successful.\
                       It highlights the commonness or popularity of coffee and motor oil, possibly implying that combining them or finding a connection between them could be beneficial.\
                       Step 2. Utterance (2): 'A match made in heaven.'\
                       This statement suggests a strong positive emotional response or approval.\
                       It portrays a sense of perfect or ideal compatibility between two things, indicating a highly positive outcome or result.\
                       Based on the analysis, there is an emotional cause-and-effect relationship between utterance (1) and utterance (2). The optimistic, hopeful tone in utterance (1) influences the strong positive emotional response in utterance (2), as it expresses the belief that the combination of coffee and motor oil is an excellent match. \
                       ''''''\
                       Based on the above Example, re-answer the question: " + prompt_no_implicit
        messages_no_implicit.append({"role": "user", "content": prompt_iter})

        result['Q2'] = prompt_no_implicit

        completion_no_implicit = client.chat.completions.create(
            model=MODEL,
            messages=messages_no_implicit,
            max_tokens=MAX_TOKENS,
            seed=SEED,
            )
        chat_response_no_implicit = completion_no_implicit
        answer_no_implicit = chat_response_no_implicit.choices[0].message.content
        result['A2'] = answer_no_implicit
        messages_no_implicit.append({"role": "assistant", "content": answer_no_implicit})
        print(f'ChatGPT: {answer_no_implicit}')
        
        if true_label == 1:
            if "No" in answer_no_implicit or "doesn't" in answer_no_implicit or "is not" in answer_no_implicit or "dose not" in answer_no_implicit:
                error += 1
                result['result'] = '{}: GPT预测错误！错误类别：存在性问题。'.format(PROMPT)
                error_type['存在性'] += 1
                judge['result'] = 'ERROR(0)'
                print('{}: GPT预测错误！错误类别：存在性问题。'.format(PROMPT))
            elif "Yes" in answer_no_implicit:
                prompt_no_implicit2 = "Judge the direction of the relationship:\
                                        Dose the utterance (1) causes the utterance (2) or Dose the utterance (2) causes the utterance (1) ?"
                messages_no_implicit.append({"role": "user", "content": prompt_no_implicit2})

                result['Q2'] = prompt_no_implicit2

                completion_no_implicit = client.chat.completions.create(
                    model=MODEL,
                    messages=messages_no_implicit,
                    max_tokens=MAX_TOKENS,
                    seed=SEED,
                )
                chat_response_no_implicit = completion_no_implicit
                answer_no_implicit = chat_response_no_implicit.choices[0].message.content

                result['A2'] = answer_no_implicit

                print(f'ChatGPT: {answer_no_implicit}')
                messages_no_implicit.append({"role": "assistant", "content": answer_no_implicit})

                if "Utterance (2) causes utterance (1)" in answer_no_implicit or "utterance (2) causes utterance (1)" in answer_no_implicit or "Utterance (2) causes Utterance (1)" in answer_no_implicit:
                    error += 1
                    result['result'] = '{}: GPT预测错误！错误类别：方向性问题。'.format(PROMPT)
                    print('{}:GPT预测错误！错误类别：方向性问题'.format(PROMPT))
                    error_type['方向性'] += 1
                    judge['result'] = 'ERROR(1)'
                elif "Utterance (1) causes utterance (2)" in answer_no_implicit or "utterance (1) causes utterance (2)" in answer_no_implicit or "Utterance (1) causes Utterance (2)" in answer_no_implicit:
                    result['result'] = '{}:GPT预测正确！'.format(PROMPT)
                    print('{}:GPT预测正确！'.format(PROMPT))
                    judge['result'] = 'CORRECT'
                else: #很大概率是gtp回答了：不确定的结果
                    #raise Exception('GPT_no_implicit 出错了！')
                    error += 1
                    result['result'] = '{}:GPT预测错误！错误类别：方向性问题。'.format(PROMPT)
                    error_type['方向性'] += 1
                    judge['result'] = 'ERROR(0)'
                    print('{}:GPT预测错误！错误类别：方向性问题'.format(PROMPT))         
            else:
                #raise Exception('GPT_no_implicit 出错了！')
                error += 1
                result['result'] = '{}: GPT预测错误！错误类别：存在性问题。'.format(PROMPT)
                error_type['存在性'] += 1
                judge['result'] = 'ERROR(0)'
                print('{}: GPT预测错误！错误类别：存在性问题。'.format(PROMPT))
        else:
            if "Yes" in answer_no_implicit:
                error += 1
                result['result'] = '{}: GPT预测错误！'.format(PROMPT)
                judge['result'] = 'ERROR(-1)'
                print('{}: GPT预测错误！'.format(PROMPT))
            elif "No" in answer_no_implicit or "doesn't" in answer_no_implicit or "is not" in answer_no_implicit or "dose not" in answer_no_implicit:
                result['result'] = '{}:GPT预测正确！'.format(PROMPT)
                print('{}:GPT预测正确！'.format(PROMPT))
                judge['result'] = 'CORRECT'   
            else:
                error += 1
                result['result'] = '{}: GPT预测错误！'.format(PROMPT)
                judge['result'] = 'ERROR(-1)'
                print('{}: GPT预测错误！'.format(PROMPT))
    
    except Exception as e:
        error_number += 1
        print('No.{} pair Exception! \n the type of Exception is:{}'.format(str(num+1),e))  
        error = y1
        error_type = y2
        continue  
    ALL_result.append(judge) 
    all_result.append(result)
          
with open('Result_{}.json'.format(PROMPT),'w') as f:
    json.dump(ALL_result,f) 
with open('detail_Result_{}.json'.format(PROMPT),'w') as f:
    json.dump(all_result,f) 
