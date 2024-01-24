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
PROMPT = 'our-method'  # our-method(zero-shot-cot)  random 
all_result = []
ALL_result = []

from datetime import datetime
current_datetime = datetime.now()
current_date = current_datetime.date()
dir1 = PROMPT + "_result_"+'epoch'+str(EPOCHS)+'_pair'+str(NUM_PAIRS)+'_'+str(current_date)
os.makedirs(dir1, exist_ok=True)

error_no_implicit = 0
error_type_no_implicit = {'存在性':0,'方向性':0}
error_implicit = {}
for i in range(EPOCHS+1):
    error_implicit[i] = 0
error_type_implicit = {}
for i in range(EPOCHS+1):
    error_type_implicit[i] = {'存在性':0,'方向性':0}
error_compute = {}
for i in range(EPOCHS+1):
    error_compute[i] = 0
error_number = 0

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
client = OpenAI()
client.api_key = os.environ.get("OPENAI_API_KEY")
print("OpenAI API Key:", os.environ.get("OPENAI_API_KEY"))

def split_answer(str):
  implicit_causes = []
  sentences = re.findall(r'\d+\.\s*([^.!?]*[.!?])', str)
  for idx, sentence in enumerate(sentences, start=1):
      implicit_causes.append(sentence.strip())
  return implicit_causes

def tensor_to_str(t):
  s = str(t[0].numpy().tolist()).strip('[').strip(']')
  s = "(" + s + ")"
  return s

def tensor_to_str2(t):
  t = t.numpy().tolist()
  dict = {'1':'first','2':'second','3':'third','4':'fourth','5':'fifth','6':'sixth','7':'seventh','8':'eighth','9':'ninth','10':'tenth'}
  s = " "
  for i in t[0][:4]:
     s = s + dict[str(i).strip()] + ','
  s = s.strip(',')
  return s

def cos_sim(A,B,AI,BI):
  A = model.encode(A)
  B = model.encode(B)
  AI = model.encode(AI)
  BI = model.encode(BI)
  
  cos_A_AI = []
  cos_A_BI = []
  cos_B_AI = []
  cos_B_BI = []
  cos_A__B_AI = []
  cos_B__A_BI = []
  for i in AI:
      cos_A_AI.append(util.pytorch_cos_sim(i,A)[0].numpy())
      cos_B_AI.append(util.pytorch_cos_sim(i,B)[0].numpy())
      cos_A__B_AI.append(util.pytorch_cos_sim(A,B+i)[0].numpy())
  for j in BI:
      cos_A_BI.append(util.pytorch_cos_sim(j,A)[0].numpy())
      cos_B_BI.append(util.pytorch_cos_sim(j,B)[0].numpy())
      cos_B__A_BI.append(util.pytorch_cos_sim(A +j,B)[0].numpy())

  cos_A_AI = torch.tensor(np.array(cos_A_AI)).view(1,-1)
  cos_A_BI = torch.tensor(np.array(cos_A_BI)).view(1,-1)
  cos_B_AI = torch.tensor(np.array(cos_B_AI)).view(1,-1)
  cos_B_BI = torch.tensor(np.array(cos_B_BI)).view(1,-1)
  cos_A__B_AI = torch.tensor(np.array(cos_A__B_AI)).view(1,-1)
  cos_B__A_BI = torch.tensor(np.array(cos_B__A_BI)).view(1,-1)

  evalA = 2 - cos_A__B_AI + cos_B_AI
  evalB = 2 - cos_B__A_BI + cos_A_BI

  evalA2 = cos_A_AI - cos_A__B_AI
  evalB2 = cos_B_BI - cos_B__A_BI

  vals1, indices1 = cos_A_AI.topk(k=10, dim=1, largest=True, sorted=True)
  vals2, indices2 = cos_A__B_AI.topk(k=10, dim=1, largest=True, sorted=True)
  vals3, indices3 = cos_B_AI.topk(k=10, dim=1, largest=True, sorted=True)
  vals4, indices4 = cos_B_BI.topk(k=10, dim=1, largest=True, sorted=True)
  vals5, indices5 = cos_B__A_BI.topk(k=10, dim=1, largest=True, sorted=True)
  vals6, indices6 = cos_A_BI.topk(k=10, dim=1, largest=True, sorted=True)

  #valsA, indicesA = evalA.topk(k=10, dim=1, largest=False, sorted=True)
  #valsB, indicesB = evalB.topk(k=10, dim=1, largest=False, sorted=True)

  valsA2, indicesA2 = evalA2.topk(k=10, dim=1, largest=False, sorted=True)
  valsB2, indicesB2 = evalB2.topk(k=10, dim=1, largest=False, sorted=True)

  rankA1 = tensor_to_str(indices2+1) 
  rankA2 = tensor_to_str(indices3+1)
  rankB1 = tensor_to_str(indices5+1) 
  rankB2 = tensor_to_str(indices6+1)
  
  rankA = tensor_to_str(indicesA2+1)
  rankB = tensor_to_str(indicesB2+1)
  Ascore = torch.mean(vals2) - torch.mean(vals1)
  Bscore = torch.mean(vals5) - torch.mean(vals4)
  return rankA1,rankA2,rankB1,rankB2,rankA,rankB,Ascore.numpy(),Bscore.numpy()

with open('dailydialog_train.json','r') as f:
  data = json.load(f)
#data[0]={'A': 'Guess who I saw yesterday ?', 'B': 'Avril Lavigen !', 'A_emotion': 'neutral', 'B_emotion': 'happiness'}

for i in range(len(data)): 
    y1 = error_no_implicit
    y2 = error_compute
    y3 = error_implicit
    y4 = error_type_no_implicit
    y5 = error_type_implicit
    try:
        num = i
        result = {} 
        judge = {} 
        
        print('\n================NO.{} sample ================\n'.format(str(i+1)))
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

        #no implicit
        messages_no_implicit = []
        prompt_no_implicit = "Here are two utterances: (1) " + A + "(2) " + B + " \
                               Is there a emotional cause-and-effect relationship between utterance (1) and utterance (2)?  Let's think step by step."
        messages_no_implicit.append({"role": "user", "content": prompt_no_implicit})

        result['Q_no_implicit'] = prompt_no_implicit

        completion_no_implicit = client.chat.completions.create(
            model=MODEL,
            messages=messages_no_implicit,
            max_tokens=MAX_TOKENS,
            seed=SEED,
            )
        chat_response_no_implicit = completion_no_implicit
        answer_no_implicit = chat_response_no_implicit.choices[0].message.content
        result['A_no_implicit'] = answer_no_implicit

        messages_no_implicit.append({"role": "assistant", "content": answer_no_implicit})
        messages_no_implicit.append({"role": "user", "content": "Therefore, the answer is (Answer Yes or No directly)"})

        completion_no_implicit = client.chat.completions.create(
            model=MODEL,
            messages=messages_no_implicit,
            max_tokens=MAX_TOKENS,
            seed=SEED,
            )
        chat_response_no_implicit = completion_no_implicit
        answer_no_implicit = chat_response_no_implicit.choices[0].message.content

        print(f'ChatGPT: {answer_no_implicit}')
        messages_no_implicit.append({"role": "assistant", "content": answer_no_implicit})

        if true_label == 1:
            if "No" in answer_no_implicit:
                error_no_implicit += 1
                result['result_no_implicit'] = 'zero-shot-cot: GPT预测错误！错误类别：存在性问题。'
                error_type_no_implicit['存在性'] += 1
                judge['result_no_implicit'] = 'ERROR(0)'
                print('zero-shot-cot: GPT预测错误！错误类别：存在性问题。')
            elif "Yes" in answer_no_implicit:
                # prompt_no_implicit2 = "Judge the direction of the causal relationship: \
                #                         Dose utterance (1) causes utterance (2) or Dose utterance (2) causes utterance (1) ?\
                #                         Answer 'Utterance (1) causes utterance (2)' or 'Utterance (2) causes utterance (1)' directly"
                prompt_no_implicit2 = "Judge the direction of the relationship:\
                                        Dose utterance (1) causes utterance (2) or Dose utterance (2) causes utterance (1) ?"
                messages_no_implicit.append({"role": "user", "content": prompt_no_implicit2})

                result['Q_no_implicit2'] = prompt_no_implicit2

                completion_no_implicit = client.chat.completions.create(
                    model=MODEL,
                    messages=messages_no_implicit,
                    max_tokens=MAX_TOKENS,
                    seed=SEED,
                )
                chat_response_no_implicit = completion_no_implicit
                answer_no_implicit = chat_response_no_implicit.choices[0].message.content

                result['A_no_implicit2'] = answer_no_implicit

                print(f'ChatGPT: {answer_no_implicit}')
                messages_no_implicit.append({"role": "assistant", "content": answer_no_implicit})

                if "causes utterance (1)" in answer_no_implicit or "causes Utterance (1)" in answer_no_implicit:
                    error_no_implicit += 1
                    result['result_no_implicit'] = 'zero-shot-cot: GPT预测错误！错误类别：方向性问题。'
                    print('zero-shot-cot: GPT预测错误！错误类别：方向性问题')
                    error_type_no_implicit['方向性'] += 1
                    judge['result_no_implicit'] = 'ERROR(1)'
                
                elif "causes utterance (2)" in answer_no_implicit or "causes Utterance (2)" in answer_no_implicit:
                    result['result_no_implicit'] = 'zero-shot-cot: GPT预测正确！'
                    print('zero-shot-cot: GPT预测正确！')
                    judge['result_no_implicit'] = 'CORRECT'
                else: #很大概率是gpt回答了：不确定的结果
                    #raise Exception('GPT_no_implicit 出错了！')
                    error_no_implicit += 1
                    result['result_no_implicit'] = 'zero-shot-cot: GPT预测错误！错误类别：方向性问题。'
                    error_type_no_implicit['方向性'] += 1
                    judge['result_no_implicit'] = 'ERROR(0)'
                    print('zero-shot-cot: GPT预测错误！错误类别：方向性问题')         
            else:
                raise Exception('GPT_no_implicit 出错了！')
        else:
            if "Yes" in answer_no_implicit:
                error_no_implicit += 1
                result['result_no_implicit'] = 'zero-shot-cot: GPT预测错误！'
                error_type_no_implicit['存在性'] += 1
                judge['result_no_implicit'] = 'ERROR(-1)'
                print('zero-shot-cot: GPT预测错误！')
            elif "No" in answer_no_implicit:
                result['result_no_implicit'] = 'zero-shot-cot: GPT预测正确！'
                judge['result_no_implicit'] = 'CORRECT'
                print('zero-shot-cot: GPT预测正确！')         
            else:
                raise Exception('GPT_no_implicit 出错了！')
        
        print('\n------------Starting Iteration--------------\n')
        
        messagesA = []
        promptA = "There is an utterance \"" + A + "\". Evidently, the speaker feels " + A_emotion + ". Please list 10 implicit causes which make this emotion emerge. (such as what is the speaker's affective drive, desire, goal etc.) Each implicit cause is expressed in one sentence."
        #promptA =  "There are two utterances: (1) \"" + A + "\" (2) \"" + B + "\". Evidently, the speaker of utterance (1) feels " + A_emotion + ". Please list 10 implicit causes of utterance (1) which make this emotion emerge. (such as what is the speaker's affective drive, desire, goal etc.) Each implicit cause is expressed in one sentence."
        messagesA.append({"role": "user", "content": promptA})

        result['Q_epoch0_A'] = promptA

        completionA = client.chat.completions.create(
            model=MODEL,
            messages=messagesA,
            max_tokens=MAX_TOKENS,
            seed=SEED,
            )
        chat_responseA = completionA
        answerA = chat_responseA.choices[0].message.content

        result['A_epoch0_A'] = answerA

        messagesA.append({"role": "assistant", "content": answerA})
        AI = split_answer(answerA)

        messagesB = []
        promptB = "There is an utterance \"" + B + "\". Evidently, the speaker feels " + B_emotion + ". Please list 10 implicit causes which make this emotion emerge. (such as what is the speaker's affective drive, desire, goal etc.) Each implicit cause is expressed in one sentence."
        #promptB =  "There are two utterances: (1) \"" + A + "\" (2) \"" + B + "\". Evidently, the speaker of utterance (2) feels " + B_emotion + ". Please list 10 implicit causes of utterance (2) which make this emotion emerge. (such as what is the speaker's affective drive, desire, goal etc.) Each implicit cause is expressed in one sentence."
        
        messagesB.append({"role": "user", "content": promptB})

        result['Q_epoch0_B'] = promptB

        completionB = client.chat.completions.create(
            model=MODEL,
            messages=messagesB,
            max_tokens=MAX_TOKENS,
            seed=SEED,
            )
        chat_responseB = completionB
        answerB = chat_responseB.choices[0].message.content
        messagesB.append({"role": "assistant", "content": answerB})

        result['A_epoch0_B'] = answerB

        #print(f'ChatGPTB: {answerB}')
        BI = split_answer(answerB)

        rankA1,rankA2,rankB1,rankB2,rankA,rankB,Ascore,Bscore = cos_sim(A,B,AI,BI)
        print('epoch 0: Asocre_{:.4f}  Bscore_{:.4f}'.format(Ascore,Bscore))

        for epoch in range(EPOCHS+1):
            print('\n======================epoch:{}========================\n'.format(epoch))

            if epoch > 0:
                
                #用于 （1,2,...） the number of rank is 3   包含score rank 
                # promptAA = "The 10 implicit causes listed in the above answers are listed as " + rankA1 + " in descending order of their relevance to the original utterance \"" + A + "\" and " + rankA2 + " in descending order of their probability of being mistaken for the implicit causes of other utterances. \
                #            Taking the above conditions into consideration, the quality ranking of these 10 implicit causes is " + rankA + ". \
                #            Based on this feedback, please regenerate 10 implicit causes(such as what is the speaker's affective dirve, desire, goal etc.) which are more closely related to the original utterance, but less likely to be mistaken for other utterances.\
                #            Each implicit cause is expressed in one sentence."
                
                #用于  first、second... the number of rank is 2   不包含score rank
                # promptAA = "Among the 10 implicit causes listed in the above answers, the " + rankA1 + " implicit causes are most relevance to the original utterance \"" + A + "\" and the" + rankA2 + " implicit causes are most likely to be mistaken for the implicit causes of other utterances. \
                #            Based on this feedback, please re-generate 10 implicit causes(such as what is the speaker's affective dirve, desire, goal etc.) which are more closely related to the original utterance, but less likely to be mistaken for other utterances.\
                #            Each implicit cause is expressed in one sentence."
                
                #用于 （1,2,...） the number of rank is 1  不包含score rank和rank2 有emotion
                promptAA = "The 10 implicit causes listed in the above answers are listed as " + rankA1 + " in descending order of their relevance to the original utterance \"" + A + "\"\
                            Based on this feedback and the speaker feels " + A_emotion + ", please re-generate 10 implicit causes(such as what is the speaker's affective dirve, desire, goal etc.) which are more closely related to the original utterance, but less likely to be mistaken for other utterances, but less likely to be mistaken for other utterances.\
                            Each implicit cause is expressed in one sentence."
                
                #用于 first、second... the number of rank is 1  不包含score rank和rank2 有emotion
                # promptAA = "Among the 10 implicit causes listed in the above answers, the " + rankA1 + " implicit causes are most relevance to the original utterance \"" + A + "\"\
                #             Based on this feedback and the speaker feels " + A_emotion + ", please re-generate 10 implicit causes(such as what is the speaker's affective dirve, desire, goal etc.) which are more closely related to the original utterance, but less likely to be mistaken for other utterances, but less likely to be mistaken for other utterances.\
                #             Each implicit cause is expressed in one sentence."
                
                #random
                # promptAA = "please re-generate 10 implicit causes(such as what is the speaker's affective dirve, desire, goal etc.) which are more closely related to the original utterance \"" + A + "\", but less likely to be mistaken for other utterances, but less likely to be mistaken for other utterances.\
                #              Each implicit cause is expressed in one sentence."
                
                #==========================================================================================================

                #用于 （1,2,...） the number of rank is 3 包含score rank
                # promptBB = "The 10 implicit causes listed in the above answers are listed as " + rankB1 + " in descending order of their relevance to the original utterance \"" + B + "\" and " + rankB2 + " in descending order of their probability of being mistaken for the implicit causes of other utterances. \
                #             Taking the above conditions into consideration, the quality ranking of these 10 implicit causes is " + rankB + ". \
                #             Based on this feedback, please regenerate 10 implicit causes(such as what is the speaker's affective drive, desire, goal etc.) which are more closely related to the original utterance, but less likely to be mistaken for other utterances.\
                #             Each implicit cause is expressed in one sentence."

                #用于  first、second... the number of rank is 2   不包含score rank
                # promptBB = "Among the 10 implicit causes listed in the above answers, the " + rankB1 + " implicit causes are most relevance to the original utterance \"" + B + "\" and the" + rankB2 + " implicit causes are most likely to be mistaken for the implicit causes of other utterances. \
                #            Based on this feedback, please regenerate 10 implicit causes(such as what is the speaker's affective drive, desire, goal etc.) which are more closely related to the original utterance, but less likely to be mistaken for other utterances.\
                #            Each implicit cause is expressed in one sentence."
                
                #用于 （1,2,...） the number of rank is 1  不包含score rank和rank2 有emotion
                promptBB = "The 10 implicit causes listed in the above answers are listed as " + rankB1 + " in descending order of their relevance to the original utterance \"" + B + "\" \
                           Based on this feedback and the speaker feels " + B_emotion + ", please regenerate 10 implicit causes(such as what is the speaker's affective drive, desire, goal etc.) which are more closely related to the original utterance, but less likely to be mistaken for other utterances, but less likely to be mistaken for other utterances.\
                           Each implicit cause is expressed in one sentence."

                #用于 first、second... the number of rank is 1  不包含score rank和rank2 有emotion
                # promptBB = "Among the 10 implicit causes listed in the above answers, the " + rankB1 + " implicit causes are most relevance to the original utterance \"" + B + "\"\
                #             Based on this feedback and the speaker feels " + B_emotion + ", please regenerate 10 implicit causes(such as what is the speaker's affective drive, desire, goal etc.) which are more closely related to the original utterance, but less likely to be mistaken for other utterances, but less likely to be mistaken for other utterances.\
                #             Each implicit cause is expressed in one sentence."
                
                #random
                # promptBB = "please re-generate 10 implicit causes(such as what is the speaker's affective dirve, desire, goal etc.) which are more closely related to the original utterance \"" + B + "\", but less likely to be mistaken for other utterances, but less likely to be mistaken for other utterances.\
                #              Each implicit cause is expressed in one sentence."
                
                messagesA.append({"role": "user", "content": promptAA})
                messagesB.append({"role": "user", "content": promptBB})

                a = 'Q_epoch'+str(epoch)+'_A'
                b = 'Q_epoch'+str(epoch)+'_B'
                result[a] = promptAA
                result[b] = promptBB

                completionA = client.chat.completions.create(
                    model=MODEL,
                    messages=messagesA,
                    max_tokens=MAX_TOKENS,
                    seed=SEED,
                )

                chat_responseA = completionA
                answerA = chat_responseA.choices[0].message.content
                #print(f'ChatGPTA: {answerA}')
                AI = split_answer(answerA)
                del messagesA[0]
                del messagesA[0]
                messagesA.append({"role": "assistant", "content": answerA})

                completionB = client.chat.completions.create(
                    model=MODEL,
                    messages=messagesB,
                    max_tokens=MAX_TOKENS,
                    seed=SEED,
                )

                chat_responseB = completionB
                answerB = chat_responseB.choices[0].message.content
                BI = split_answer(answerB)
                #print(f'ChatGPTB: {answerB}')
                del messagesB[0]
                del messagesB[0]
                messagesB.append({"role": "assistant", "content": answerB})

                rankA1,rankA2,rankB1,rankB2,rankA,rankB,Ascore,Bscore = cos_sim(A,B,AI,BI)
                print('epoch {}: Asocre_{:.4f}  Bscore_{:.4f}'.format(epoch,Ascore,Bscore))

                a = 'A_epoch'+str(epoch)+'_A'
                b = 'A_epoch'+str(epoch)+'_B'
                result[a] = answerA
                result[b] = answerB
        
            #------------------------------------compute---------------------------------------
            # m = 'result_compute_epoch'+str(epoch)
            # if true_label == 1:
            #     if (Ascore < 0 and Bscore < 0) or (Ascore > 0 and Bscore < 0):
            #         error_compute[epoch] += 1
            #         result[m] = "通过判定条件，结果预测错误！"
            #         judge[m] = 'ERROR'
            #         print("COMPUTE: 通过判定条件，结果预测错误！\n")
            #     elif Ascore < 0 and Bscore > 0:
            #         result[m] = '通过判定条件，结果预测正确！'
            #         judge[m] = 'CORRECT'
            #         print("COMPUTE: 通过判定条件，结果预测正确！\n")
            #     else:
            #         if Ascore < Bscore:
            #             result[m] = '通过判定条件，结果预测正确！'
            #             judge[m] = 'CORRECT'
            #             print("COMPUTE: 通过判定条件，结果预测正确！\n")
            #         else:
            #             error_compute[epoch] += 1
            #             result[m] = "通过判定条件，结果预测错误！"
            #             judge[m] = 'ERROR'
            #             print("COMPUTE: 通过判定条件，结果预测错误！\n")
            # else:
            #     if (Ascore < 0 and Bscore > 0) or (Ascore > 0 and Bscore < 0):
            #         error_compute[epoch] += 1
            #         result[m] = "通过判定条件，结果预测错误！"
            #         judge[m] = 'ERROR'
            #         print("COMPUTE: 通过判定条件，结果预测错误！\n")
            #     else:
            #         result[m] = '通过判定条件，结果预测正确！'
            #         judge[m] = 'CORRECT'
            #         print("COMPUTE: 通过判定条件，结果预测正确！\n")
            
            #------------------------------------implicit---------------------------------------
            # extract the best AI,BI
            idxA = int(rankA.strip('(').strip(')').split(',')[0]) - 1
            idxB = int(rankB.strip('(').strip(')').split(',')[0]) - 1
            # extract the AI,BI randomly
            # random_integer = random.randint(0, 9)    
            # idxA = int(rankA.strip('(').strip(')').split(',')[random_integer]) - 1
            # idxB = int(rankB.strip('(').strip(')').split(',')[random_integer]) - 1
            AI = AI[idxA]
            BI = BI[idxB]

            a = 'bestA_epoch'+str(epoch)
            b = 'bestB_epoch'+str(epoch)
            result[a] = AI
            result[b] = BI

            messages_implicit = []
            prompt_implicit = "Here are two utterances and their implicit causes for speakers to say them. \
                                Utterance (1): " + A + "The implicit cause of utterance (1) is \"" + AI + "\" \
                                Utterance (2): " + B + "The implicit cause of utterance (2) is \"" + BI + "\" \
                                Consider about the content of utterances and their implicit causes, is there a emotional cause-and-effective relationship between the utterance (1) and utterance (2)? Let's think step by step."
            messages_implicit.append({"role": "user", "content": prompt_implicit})

            m = 'Q_implicit_epoch'+str(epoch)
            result[m] = prompt_implicit

            completion_implicit = client.chat.completions.create(
                model=MODEL,
                messages=messages_implicit,
                max_tokens=MAX_TOKENS,
                seed=SEED,
                )
            chat_response_implicit = completion_implicit
            answer_implicit = chat_response_implicit.choices[0].message.content
            m = 'A_implicit_epoch'+str(epoch)
            result[m] = answer_implicit

            messages_implicit.append({"role": "assistant", "content": answer_implicit})
            messages_implicit.append({"role": "user", "content": "Therefore, the answer is (Answer Yes or No directly)"})

            completion_implicit = client.chat.completions.create(
                model=MODEL,
                messages=messages_implicit,
                max_tokens=MAX_TOKENS,
                seed=SEED,
                )
            chat_response_implicit = completion_implicit
            answer_implicit = chat_response_implicit.choices[0].message.content
            messages_implicit.append({"role": "assistant", "content": answer_implicit})

            #m = 'A_implicit_epoch'+str(epoch)
            #result[m] = answer_implicit
            print(f'ChatGPT: {answer_implicit}')

            m = 'result_implicit_epoch'+str(epoch)
            if true_label == 1:
                if "No" in answer_implicit:
                    error_implicit[epoch] += 1
                    error_type_implicit[epoch]['存在性'] += 1
                    result[m] = 'our method: GPT预测错误！错误类别：存在性问题。'
                    judge[m] = 'ERROR(0)'
                    print('our method: GPT预测错误！错误类别：存在性问题')
                elif "Yes" in answer_implicit:
                    prompt_implicit2 = "Judge the direction of the relationship: \
                                        Dose utterance (1) causes utterance (2) or Dose utterance (2) causes utterance (1) ?"
                    messages_implicit.append({"role": "user", "content": prompt_implicit2})

                    n = 'Q_implicit2_epoch'+str(epoch)
                    nn = 'A_implicit2_epoch'+str(epoch)
                    result[n] = prompt_implicit2

                    completion_implicit = client.chat.completions.create(
                        model=MODEL,
                        messages=messages_implicit,
                        max_tokens=MAX_TOKENS,
                        seed=SEED,
                    )
                    chat_response_implicit = completion_implicit
                    answer_implicit = chat_response_implicit.choices[0].message.content

                    result[nn] = answer_implicit

                    print(f'ChatGPT: {answer_implicit}')
                    messages_implicit.append({"role": "assistant", "content": answer_implicit})

                    if "causes utterance (1)" in answer_implicit or "causes Utterance (1)" in answer_implicit:
                        error_implicit[epoch] += 1
                        error_type_implicit[epoch]['方向性'] += 1
                        result[m] = 'our method: GPT预测错误！错误类别：方向性问题。'
                        judge[m] = 'ERROR(1)'
                        print('our method: GPT预测错误！错误类别：方向性问题\n')
                    
                    elif "causes utterance (2)" in answer_implicit or "causes Utterance (2)" in answer_implicit:
                        result[m] = 'our method: GPT预测正确！'
                        judge[m] = 'CORRECT'
                        print('our method: GPT预测正确！\n')
                    else:
                        #raise Exception('GPT_implicit  出错了！')
                        error_implicit[epoch] += 1
                        error_type_implicit[epoch]['方向性'] += 1
                        result[m] = 'our method: GPT预测错误！错误类别：方向性问题。'
                        judge[m] = 'ERROR(1)'
                        print('our method: GPT预测错误！错误类别：方向性问题\n')
                else:
                    raise Exception('GPT_implicit  出错了！')
            else:
                if "Yes" in answer_implicit:
                    error_implicit[epoch] += 1
                    error_type_implicit[epoch]['存在性'] += 1
                    result[m] = 'our method: GPT预测错误！'
                    judge[m] = 'ERROR(-1)'
                    print('our method: GPT预测错误！')
                elif "No" in answer_implicit:
                    result[m] = 'our method: GPT预测正确！'
                    judge[m] = 'CORRECT'
                    print('our method: GPT预测正确！\n')
                else:
                    raise Exception('GPT_implicit  出错了！')

        all_result.append(result)
        ALL_result.append(judge)

        with open('{}/pair_{}_{}.txt'.format(dir1,str(num+1),str(true_label)),'w',encoding='utf8') as f:
            f.write('A: {}\nB: {}\n\n'.format(result['A'],result['B']))
            f.write('----------------------------------------------------------------\n\n')
            if 'Q_no_implicit2' in result:
                f.write('NO implicit:\nQ: {}\nA: {}\n\nQ: {}\nA: {}\n\nResult: {}\n\n'.format(result['Q_no_implicit'],result['A_no_implicit'],result['Q_no_implicit2'],result['A_no_implicit2'],result['result_no_implicit']))
            else:
                f.write('NO implicit:\nQ: {}\nA: {}\n\nResult: {}\n\n'.format(result['Q_no_implicit'],result['A_no_implicit'],result['result_no_implicit']))
            f.write('----------------------------------------------------------------\n\n')
            f.write('Start iteration......\n\n')
            for epoch in range(EPOCHS + 1):
                a1 = 'Q_epoch' + str(epoch) + '_A'
                b1 = 'Q_epoch' + str(epoch) + '_B'
                a2 = 'A_epoch' + str(epoch) + '_A'
                b2 = 'A_epoch' + str(epoch) + '_B'
                m = 'result_implicit_epoch'+str(epoch)
                kk = 'A_implicit_epoch'+str(epoch)
                k = 'Q_implicit_epoch'+str(epoch)
                n = 'Q_implicit2_epoch'+str(epoch)
                nn = 'A_implicit2_epoch'+str(epoch)
                f.write('--------------------------EPOCH: {} -------------------------\n\n'.format(str(epoch)))
                
                f.write('Q_A: {}\nA_A: {}\n\nQ_B: {}\nA_B: {}\n\n'.format(result[a1],result[a2],result[b1],result[b2]))

                a = 'bestA_epoch'+str(epoch)
                b = 'bestB_epoch'+str(epoch)
                f.write('BEST_A: {}\nBEST_B:{}\n\n'.format(result[a],result[b]))
                if n in result:
                    f.write('Implicit:\nQ: {}\nA: {}\n\nQ: {}\nA: {}\n\nResult: {}\n\n'.format(result[k],result[kk],result[n],result[nn],result[m]))
                else:
                    f.write('Implicit:\nQ: {}\nA: {}\n\nResult: {}\n\n'.format(result[k],result[kk],result[m]))
            f.write('End interation......\n\n')
            f.write('----------------------------------------------------------------\n\n')
        
        for i in range(EPOCHS+1):
            print('-----------------------result_epoch_{}-------------------------------'.format(i))
            print("our method: the number of error：{}次".format(error_implicit[i])) 
            #print('positive sample: the number of 【存在性】error：{} ; 【方向性】error：{}'.format(error_type_implicit[i]['存在性'],error_type_implicit[i]['方向性']))

    except Exception as e:
        error_number += 1
        print('No.{} pair Exception! \n the type of Exception is: {}'.format(str(num+1),e))  
        error_no_implicit = y1
        error_compute = y2
        error_implicit = y3
        error_type_no_implicit = y4
        error_type_implicit = y5
        continue      

with open("Result_"+'epoch'+str(EPOCHS)+'_pair'+str(NUM_PAIRS)+'_'+str(current_date)+'_'+PROMPT+'.json','w') as f:
    json.dump(ALL_result,f)

print('\n-------------End: statistical results------------\n')
print('Number of abnormal pairs: {}'.format(str(error_number)))
print("zero-shot-cot: the number of error: {}次".format(error_no_implicit)) 
#print('positive sample: the number of 【存在性】error：{} ; 【方向性】error：{}'.format(error_type_no_implicit['存在性'],error_type_no_implicit['方向性']))

for i in range(EPOCHS+1):
    print('-----------------------result_epoch_{}-------------------------------'.format(i))
    print("our method: the number of error: {}次".format(error_implicit[i])) 
    print('positive sample: the number of【存在性】error: {} ; 【方向性】error：{}'.format(error_type_implicit[i]['存在性'],error_type_implicit[i]['方向性']))
with open('allresult_{}_epoch{}_pair{}_{}.pkl'.format(MODEL,EPOCHS,NUM_PAIRS,current_date),'wb') as f:
    pkl.dump(all_result,f)
