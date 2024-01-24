import json
import random
with open('dailydialog_0.json','r') as f:
    data_0 = json.load(f)
with open('dailydialog_1.json','r') as f:
    data_1 = json.load(f)
#data[0]={'A': 'Guess who I saw yesterday ?', 'B': 'Avril Lavigen !', 'A_emotion': 'neutral', 'B_emotion': 'happiness'}

random.seed(233)
idxlist_0 = random.sample(range(0,len(data_0)),100)
idxlist_1 = random.sample(range(0,len(data_1)),100)
print(idxlist_0)
print(idxlist_1)
data = []
for idx in idxlist_0:
    data.append(data_0[idx])
for idx in idxlist_1:
    data.append(data_1[idx])
my_list = list(range(0, len(data)))
random.shuffle(my_list)
data2 = []
for i in my_list:
    data2.append(data[i])

with open('dailydialog_train.json','w') as f:
    json.dump(data2,f)