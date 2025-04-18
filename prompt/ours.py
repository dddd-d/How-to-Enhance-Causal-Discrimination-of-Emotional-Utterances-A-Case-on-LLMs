import ast
import re
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
from openai import OpenAI
import json
from sentence_transformers import SentenceTransformer,util
import random
import json
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

MODEL = "gpt-3.5-turbo" 
SEED = 233
EPOCHS = 0
PROMPT = 'Implicit-causes-enhanced' 
EVAL = 'external_eval'


TYPE = 'all' #common/chain/reverse/all
if TYPE == 'all':
    filename = 'dailydialog_test.json'

data = []
with open(filename,'r') as f:
  for line in f:
      data.append(json.loads(line))

torch.manual_seed(SEED)
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

client = OpenAI()
client.api_key = os.environ.get("OPENAI_API_KEY")

prompt_generate_E1 = '''Given two utterances:
[1] {clause1}
[2] {clause2}

Instruction:
Use common sense and experience to infer the implicit background causes for the utterance [1], which are not related to the explicit content of utterance [2].

Requirements:
List 5 reasonable implicit causes of the utterance [1].
Return in the form of a python list, such as ["The speaker recently announced their pregnancy.",\n ...]'''

prompt_generate_E = '''Instruction: Use common knowledge and experience to infer the implicit causes of the given utterance.

--Example--
Given one utterance from a daily dialogue:
Oh ? You hoped to get that job , didn't you ?

Return:
["The job in question is highly competitive or desirable, leading the speaker to inquire about the individual's hopes.",
]

--To be solved--
Given one utterance from a daily dialogue: 
{clause1}

Requirements:
List 5 reasonable implicit causes.
Return in the form of a python list.'''

PROMPT_E = prompt_generate_E

prompt_iter1 = '''Given two utterances: 
[1] {clause1}
[2] {clause2}

Feedback on previous answer:
The semantic relevance to utterance [1] is ranked from most to least: {rank1}
The semantic relevance to utterance [2] is ranked from least to most: {rank2}

Instruction:
Based on the feedback, use common sense and experience to infer the implicit background causes of the utterance [1], which are not related to the explicit content of utterance [2].

Requirements:
List 5 reasonable implicit causes of the utterance [1].
Return in the form of a python list.'''  

prompt_iter = '''Instruction: Update original implicit causes of the given utterance based on the Feedback.

Given one utterance from a daily dialogue:
{clause1}

Original implicit causes:
1. {e1}
2. {e2}
3. {e3}
4. {e4}
5. {e5}

Feedback:
Original implicit causes are listed as {rank1} in descending order of their relevance to the given utterance, {rank2} in ascending order of their relevance to other utterances.

Requirements:
List 5 reasonable implicit causes so that they are more relevant to the given utterance and less relevant to other utterances
Return in the form of a python list, such as ["The speaker recently announced their pregnancy.",\n ...]'''

Implicit_Causes_Enhanced = '''Instruction: You are an expect in emotion cause extraction. Follow the given reasoning process step by step to answer the given Question.

There is a fragment of daily conversation:
{context}

The implicit cause of [{idx1}]: {implicit_cause1}

Question: Is there a causal relationship from the original utterance [{idx1}] to [{idx2}]?

Given a reasoning process:
1. Is the implicit cause of {speaker1}'s [{idx1}] the cause of the {speaker2}'s emotions in [{idx2}]?
2. Consider the context and reasoning in the previous step, think about the Question.
3. After the reasoning is over, answer the Question with either "Yes." or "No.", if you are unclear/unsure, please answer "No.".'''

Implicit_Causes_Enhanced1 = '''Given a fragment of daily conversation:
{context}

The implicit background cause of utterance [{idx1}]: {implicit_cause1}

Follow the following reasoning process step by step to infer the question "Is there a causal relationship from the original utterance [{idx1}] to [{idx2}]?":
1. Is the implicit cause of {speaker1}'s [{idx1}] the indirect cause of the {speaker2}'s emotion in [{idx2}]?
2. Is the utterance [{idx1}] the direct cause of the {speaker2}'s emotion in [{idx2}]?
3. After the reasoning is over, answer the main question with either "Yes." or "No.", if you are unclear/unsure, please answer "No.".'''

def query(message):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=message,
        seed=SEED,
        temperature=0,
        )
    return completion.choices[0].message.content

def judge_answer(response, label):
    if label == 'pos':
        if 'Yes' in response:
            return True
        elif 'No' in response:
            return False
        else:
            print(response)
            return 'ERROR'
    else:
        if 'Yes' in response:
            return False
        elif 'No' in response:
            return True
        else:
            print(response)
            return 'ERROR'

def cos_sim(U_i, U_t, E_i):
    H_i = model.encode(U_i)
    H_t = model.encode(U_t)
    H_E_i = model.encode(E_i)
    #a = util.pytorch_cos_sim(H_i,H_E_i) #tensor([[0.1]])
    sim_i_E_i = util.pytorch_cos_sim(H_i,H_E_i)[0].numpy()[0]
    sim_t_E_i = util.pytorch_cos_sim(H_t,H_E_i)[0].numpy()[0]
    sim_i__Ei_t = util.pytorch_cos_sim(H_i,H_t+H_E_i)[0].numpy()[0]

    #score = sim_i_E_i - sim_t_E_i + sim_i__Ei_t
    score = sim_i_E_i - sim_t_E_i
    return score, sim_i_E_i, sim_t_E_i

def external_eval(U_i, U_t, E_i_list, eval = 'cos'):
    scores = []
    i_Es = []
    t_Es = []
    for e in E_i_list:
        score, i_E, t_E = cos_sim(U_i, U_t, e)
        scores.append(score)
        i_Es.append(i_E)
        t_Es.append(t_E)
    rank = np.argsort(-np.array(scores), axis=-1) #从大到小排序, return array([1, 2, 0])
    best_E = E_i_list[rank[0]]
    rank1 = np.argsort(-np.array(i_Es), axis=-1)
    rank2 = np.argsort(np.array(t_Es), axis=-1)
    return best_E, rank, rank1, rank2 

def generate_E(clause1, clause2):
    messages = []
    question = PROMPT_E.replace('{clause1}', clause1).replace('{clause2}',clause2)
    messages.append({"role": "user", "content": question})

    answer = query(messages)
    messages.append({"role": "assistant", "content": answer})
    try:
        E = ast.literal_eval(answer)
    except Exception:
        E = re.findall(r'"(.*?)"', answer)
    if len(E) < 5:
        E = re.findall(r'\d+\.\s+(.*?)(?=\n\d+\.|\n?$|$)', answer, re.DOTALL)
    if len(E) < 5:
        E = re.findall(r"'(.*?)'", answer)
    return messages, E #list

def GPT_Implicit_Causes_Enhanced(E_s, context, idx_i, idx_t, speaker_s, speaker_t):
    messages_Implicit_Causes_Enhanced = []
    question = Implicit_Causes_Enhanced.replace('{context}', context).replace('{idx1}',str(idx_i)).replace('{idx2}',str(idx_t)).replace('{implicit_cause1}', E_s).replace('{speaker2}', speaker_t).replace('{speaker1}', speaker_s)
    messages_Implicit_Causes_Enhanced.append({"role": "user", "content": question})
    answer = query(messages_Implicit_Causes_Enhanced)
    return answer

def iter_E_external_eval(rank1, rank2, E, clause1, messages):
    question = prompt_iter.replace('{clause1}',clause1).replace('{rank1}', str(rank1)).replace('{rank2}', str(rank2))
    question = question.replace('{e1}', E[0])
    question = question.replace('{e2}', E[1])
    question = question.replace('{e3}', E[2])
    question = question.replace('{e4}', E[3])
    question = question.replace('{e5}', E[4])
    messages = []
    messages.append({"role": "user", "content": question})
    answer = query(messages)
    messages.append({"role": "assistant", "content": answer})
    #E = re.findall(r'"(.*?)"', answer)
    try:
        E = ast.literal_eval(answer)
    except Exception:
        E = re.findall(r'"(.*?)"', answer)
    if len(E) < 5:
        E = re.findall(r'\d+\.\s+(.*?)(?=\n\d+\.|\n?$|$)', answer, re.DOTALL)
    if len(E) < 5:
        E = re.findall(r"'(.*?)'", answer)
    return messages, E

def train_pair(U_t, U_i, context, idx_t, idx_i, speaker_t, speaker_i, label, prompt='Implicit-causes-enhanced',eval='external_eval'):
    
    p_result = {'type':label, 'U_i':U_i,'U_t':U_t,'context': context}
    infers = []
    
    p_result['iter0'] = {}
    #initial
    message_i, E_i_list = generate_E(U_i, U_t)
    
    if eval == 'external_eval':
        best_E_i, rank_i, rank1_i, rank2_i = external_eval(U_i, U_t, E_i_list)
    else:
        raise Exception('EVAL error')
    
    p_result['iter0']['E_i'] = best_E_i

    response = GPT_Implicit_Causes_Enhanced(best_E_i, context, idx_i, idx_t, speaker_i, speaker_t)
    infer = judge_answer(response, label)
    print('infer0: {}'.format(infer))
    infers.append(infer)
    p_result['iter0']['response'] = response
    p_result['iter0']['infer'] = infer

    for epoch in range(EPOCHS):
        iter_dict = {}
        message_i, E_i_list = iter_E_external_eval(rank1_i, rank2_i, E_i_list, U_i, message_i)

        if eval == 'external_eval':
            best_E_i, rank_i, rank1_i, rank2_i = external_eval(U_i, U_t, E_i_list)
        else:
            raise Exception('EVAL error')

        iter_dict['E_i'] = best_E_i

        response = GPT_Implicit_Causes_Enhanced(best_E_i, context, idx_i, idx_t, speaker_i, speaker_t)
        infer = judge_answer(response, label)
        print('infer{0}: {1}'.format(epoch+1, infer))
        infers.append(infer)
        iter_dict['response'] = response
        iter_dict['infer'] = infer

        p_result['iter'+str(epoch+1)] = iter_dict
    return p_result, infers
    
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
for i in range(0,3):#len(data)
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
        print('===================Type: {} No.{} Pair: {} \'POS\'====================='.format(TYPE,i+1, p_idx+1))
        U_t = clauses[str(p[0])]
        U_i = clauses[str(p[1])]
        
        num = min(p[0], max(Windows+1, p[0]-p[1]+1))
        context = '\n'.join(['[{0}] {1}: '.format(p[0]-n+1, speakers[p[0]-n]) + clauses[str(p[0]-n+1)] for n in range(num, 0, -1)]) 
        
        try:
            res, infers = train_pair(U_t, U_i, context, p[0], p[1], speakers[p[0]-1], speakers[p[1]-1], 'pos', PROMPT, EVAL)

            t_label.append([1, 0])
            if infers[0]:
                p_label.append([1, 0])
            else:
                p_label.append([0, 1])

            result.append(res)
        except Exception as e:
            print(e)
            pass

    for p_idx in range(len(neg)):
        p = neg[p_idx]
        print('===================Type: {} No.{} Pair: {} \'NEG\'====================='.format(TYPE,i+1, p_idx+1))
        U_t = clauses[str(p[0])]
        U_i = clauses[str(p[1])]
        
        num = min(p[0], max(Windows+1, p[0]-p[1]+1))
        context = '\n'.join(['[{0}] {1}: '.format(p[0]-n+1, speakers[p[0]-n]) + clauses[str(p[0]-n+1)] for n in range(num, 0, -1)]) 
        
        try:
            res, infers = train_pair(U_t, U_i, context, p[0], p[1], speakers[p[0]-1], speakers[p[1]-1],'neg', PROMPT,EVAL)

            t_label.append([0, 1])
            if infers[0]:
                p_label.append([0, 1])
            else:
                p_label.append([1, 0])

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
    
    with open('outputs/{0}/gpt3.5_{1}_{2}_{3}_{4}.json'.format(PROMPT, TYPE, PROMPT, EVAL, str(i)),'w') as f:
        for line in result:
            json.dump(line, f)
            f.write('\n')

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
