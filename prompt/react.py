import json
import random
import numpy as np
import openai
import re
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

client = OpenAI()
client.api_key = os.environ.get("OPENAI_API_KEY")
MODEL = 'gpt-3.5-turbo-ca'
SEED = 42

def query(message):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=message,
        seed=SEED,
        temperature=0,
        )
    return completion.choices[0].message.content

def react_causal_analysis(dialogue, x, y):
    messages = [
            {"role": "system", "content": "You are an expert in causal analysis. Please analyze the dialogue fragment step by step strictly according to the ReAct paradigm. Use the following format:\n\nThought: <Thinking steps>\nAction: <Execution action>\nObservation: <Action feedback>\nReflection: <Reflection and judgment on whether the logic needs to be adjusted>"},
            {"role": "user", "content": f"""Dialogue fragment：\n{dialogue}\n\nQuestion：Is there a causal relationship from the utterance [{x}] to [{y}]?"""}]
    
    max_steps = 3  # 防止无限循环
    final_answer = None
    for _ in range(max_steps):
        response = query(messages)
        
        messages.append({"role": "assistant", "content": response})

        # 检查是否可以得出最终结论
        messages.append({"role": "user", "content": "Based on your reasoning so far, can you determine the final answer? Use the following format:\n\nFinal Answer: <Answer text or \"UNCERTAIN\">"})
        response = query(messages)
        
        # 检查是否已得出最终结论
        if "UNCERTAIN" not in response:
            final_answer = re.search(r"Final Answer:\s*(.*)", response).group(1)
            break
        
        #继续分析
        messages = messages[:-1]
        messages.append({"role": "user", "content": "Continue to the next round of analysis."})
    
    return final_answer
def compute_metric(true_label, pred_label):
    tl = np.array(true_label)
    pl = np.array(pred_label)

    # 获取类别索引
    true_idx = np.argmax(tl, axis=-1)
    pred_idx = np.argmax(pl, axis=-1)

    accuracy = accuracy_score(true_idx, pred_idx)
    f1 = f1_score(tl, pl, average=None)
    p = precision_score(tl, pl, average=None)
    r = recall_score(tl, pl, average=None)

    return accuracy, f1, p, r 


Windows = 4
true_label = []
pred_label = []
random.seed(SEED)

filename = 'dailydialog_test.json'
data = []
with open(filename,'r') as f:
  for line in f:
      data.append(json.loads(line))

for i in range(3,4):#len(data)
    result = []
    t_label = []
    p_label = []

    d = data[i]
    clauses = d['clauses'] #dict
    speakers = d['speakers']
    pos = d['pos']
    neg = d['neg']

    snum = min(len(pos), len(neg))
    neg = random.sample(neg, snum)

    labels = d['labels']
    
    for p_idx in range(len(pos)):
        p = pos[p_idx]
        print('===================No.{} Pair: {} \'POS\'====================='.format(i+1, p_idx+1))
        U_t = clauses[str(p[0])]
        U_i = clauses[str(p[1])]
        
        num = min(p[0], max(Windows+1, p[0]-p[1]+1))
        context = '\n'.join(['[{0}] {1}: '.format(p[0]-n+1, speakers[p[0]-n]) + clauses[str(p[0]-n+1)] for n in range(num, 0, -1)]) 
        
        try:
            res = react_causal_analysis(context, p[0], p[1])
            print(res)
            t_label.append([1, 0])
            if res == None:
                p_label.append([0, 1])
            else:
                if 'No' in res or 'here is no' in res or 'NO' in res:
                    p_label.append([0, 1])
                    print('False')
                else:
                    p_label.append([1, 0])
                    print('True')

            result.append(res)
        except Exception as e:
            print(e)
            pass

    for p_idx in range(len(neg)):
        p = neg[p_idx]
        print('===================No.{} Pair: {} \'NEG\'====================='.format(i+1, p_idx+1))
        U_t = clauses[str(p[0])]
        U_i = clauses[str(p[1])]
        
        num = min(p[0], max(Windows+1, p[0]-p[1]+1))
        context = '\n'.join(['[{0}] {1}: '.format(p[0]-n+1, speakers[p[0]-n]) + clauses[str(p[0]-n+1)] for n in range(num, 0, -1)]) 
        
        try:
            res = react_causal_analysis(context, p[0], p[1])
            print(res)

            t_label.append([0, 1])
            if res == None:
                p_label.append([1, 0])
            else:
                if 'No' in res or 'here is no' in res or 'NO' in res:
                    p_label.append([0, 1])
                    print('True')
                else:
                    p_label.append([1, 0])
                    print('False')

            result.append(res)
        except Exception as e:
            print(e)
            pass  
    
    acc, f1, pre, rec = compute_metric(t_label, p_label)
    print('acc:', acc, 'f1:', f1.mean(), 'pre:', pre.mean(), 'rec:', rec.mean())
    result.append({'acc': acc, 'pf1': f1[0], 'nf1': f1[1],'f1': f1.mean(),'ppre': pre[0], 'npre': pre[1], 'pre': pre.mean(),'prec': rec[0], 'nrec': rec[1], 'rec': rec.mean()})

    true_label += t_label
    pred_label += p_label

    acc, f1, pre, rec = compute_metric(true_label, pred_label)
    print('现阶段:\nacc:', acc, '\nf1:', f1[0], f1[1], f1.mean(), '\nprecision:', pre[0], pre[1], pre.mean(), '\nrecall:', rec[0], rec[1], rec.mean())

true_label = np.array(true_label)
pred_label = np.array(pred_label)

# 获取类别索引
true_idx = np.argmax(true_label, axis=-1)
pred_idx = np.argmax(pred_label, axis=-1)

accuracy = accuracy_score(true_idx, pred_idx)
f1 = f1_score(true_label, pred_label, average=None)
p = precision_score(true_label, pred_label, average=None)
r = recall_score(true_label, pred_label, average=None)
print(accuracy)
print(f1[0], f1[1], f1.mean())    
print(p[0], p[1], p.mean())    
print(r[0], r[1], r.mean())  

