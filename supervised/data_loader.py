import sys
import torch
from torch.utils.data import Dataset
from utils import *
from torch.nn.utils.rnn import pad_sequence
import scipy.sparse as sp
from transformers import RobertaTokenizer


def build_train_data(fold_id, batch_size,data_type,args,shuffle=True):
    train_dataset = MyDataset(fold_id, data_type='train',args=args)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               collate_fn=train_dataset.collate_fn,shuffle=shuffle)
    return train_loader

def build_inference_data(fold_id, batch_size,args,data_type):
    dataset = MyDataset( fold_id, data_type=data_type,args=args)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                               collate_fn=dataset.collate_fn,shuffle=False)
    return data_loader

class MyDataset(Dataset):
    def __init__(self,fold_id,data_type,args):
        self.data_type=data_type
        if args.iemocaptest == True and data_type=='test':
            self.data_dir='data/iemocap/iemocap_test_cls.json'
            self.speaker_vocab=pickle.load(open('data/iemocap/speaker_vocab.pkl', 'rb'))
            self.label_vocab=pickle.load(open('data/iemocap/label_vocab.pkl' , 'rb'))
        else:
            self.data_dir='data/dailydialog/fold%s/dailydialog_%s_cls.json'%(fold_id,data_type)
            self.speaker_vocab={'stoi': {'A': 0, 'B': 1}, 'itos': ['A', 'B']}
            self.label_vocab ={'stoi': {'neutral': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}, 'itos': ['neutral', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']}
        self.args=args
        self.bert_tokenizer = RobertaTokenizer.from_pretrained(self.args.roberta_pretrain_path,local_files_only=True)
        
        
        ###从文档中读出的各种数据
        self.doc_id_list,self.doc_len_list,self.doc_couples_list,self.doc_speaker_list,self.doc_cls_list, \
        self.y_emotions_list,self.y_causes_list,self.doc_text_list,self.doc_emotion_category_list, \
        self.doc_emotion_token_list,self.bert_token_idx_list,self.bert_clause_idx_list, \
           self.bert_segments_idx_list,self.bert_token_lens_list= self.read_data_file(self.data_dir)
    
    def __len__(self):
        return len(self.y_emotions_list)
    
    def __getitem__(self, idx):
        doc_id,doc_len,doc_couples,y_emotions,y_causes=self.doc_id_list[idx],self.doc_len_list[idx],self.doc_couples_list[idx],self.y_emotions_list[idx],self.y_causes_list[idx]
        doc_speaker,doc_text,doc_cls=self.doc_speaker_list[idx],self.doc_text_list[idx],self.doc_cls_list[idx]
        doc_emotion_category,doc_emotion_token=self.doc_emotion_category_list[idx],self.doc_emotion_token_list[idx]
        bert_token_idx, bert_clause_idx = self.bert_token_idx_list[idx], self.bert_clause_idx_list[idx]
        bert_segments_idx, bert_token_lens = self.bert_segments_idx_list[idx], self.bert_token_lens_list[idx]

        if bert_token_lens > 512 and self.args.withbert==True: #文档总token数>512
        #训练方式：FULL-SENTENCES：输入只有一部分（而不是两部分），来自同一个文档或者不同文档的连续多个句子，token 总数不超过 512 。
        # 输入可能跨越文档边界，如果跨文档，则在上一个文档末尾添加文档边界token 。预训练不包含 NSP（Next Sentence Prediction）任务。只针对MLM任务
            bert_token_idx, bert_clause_idx, \
            bert_segments_idx, bert_token_lens, \
            doc_couples, y_emotions, y_causes, doc_len,doc_speaker,doc_emotion_category = \
                self.token_trunk(bert_token_idx, bert_clause_idx,bert_segments_idx, bert_token_lens,
                                doc_couples, y_emotions, y_causes, doc_len,doc_speaker,doc_emotion_category)

        bert_token_idx = torch.LongTensor(bert_token_idx)
        bert_segments_idx = torch.LongTensor(bert_segments_idx)
        bert_clause_idx = torch.LongTensor(bert_clause_idx)#bert_clause_idx:子句token开始的位置
        assert doc_len == len(y_emotions)
        return doc_id,doc_len,doc_couples,y_emotions,y_causes,doc_speaker,doc_text,doc_cls,doc_emotion_category,doc_emotion_token, \
            bert_token_idx, bert_segments_idx, bert_clause_idx, bert_token_lens
        
    def read_data_file(self, data_dir):
        datafile=data_dir
        doc_id_list = []
        doc_len_list = []
        doc_couples_list = []
        y_emotions_list, y_causes_list = [], []
        doc_speaker_list=[]
        doc_cls_list=[]
        doc_text_list=[]
        doc_emotion_category_list = []
        doc_emotion_token_list=[]
        bert_token_idx_list = []
        bert_clause_idx_list = []
        bert_segments_idx_list = []
        bert_token_lens_list = []
        data=read_json(datafile)
        for doc in data:
            doc_id=doc['doc_id']#文档id
            doc_len=doc['doc_len']#文档对话句子数量
            doc_couples = doc['pairs']#文档中的情感原因对
            doc_emotions, doc_causes = zip(*doc_couples)#将情感原因对拆成情感和原因 范围【1-14】
            doc_clauses = doc['clauses']
            doc_speaker=doc['speakers']
            doc_cls=doc['cls']
            doc_id_list.append(doc_id)###文档id汇总
            doc_len_list.append(doc_len)###文档中每个对话长度汇总
            doc_couples = list(map(lambda x: list(x), doc_couples))
            doc_couples_list.append(doc_couples)###文档中每个对话标签对汇总
            
            doc_cls_list.append(doc_cls)###文档中预训练特征汇总
            
            y_emotions, y_causes = [], []
            
            doc_str = ''
            doc_text = []
            doc_emotion_category = []
            doc_emotion_token=[]
            doc_str=''
            for i in range(doc_len): #循环每一个子句
                emotion_label = int(i + 1 in doc_emotions) #0/1 判断子句是否为情感子句
                cause_label = int(i + 1 in doc_causes) #0/1
                y_emotions.append(emotion_label)
                y_causes.append(cause_label)
                doc_speaker[i]=self.speaker_vocab['stoi'][doc_speaker[i]]    #记录子句的speaker 只有两个说话者A/B-> 0/1           
                clause_id = doc_clauses[i]['clause_id']#文档中子句的id
                assert int(clause_id) == i + 1 #确保clause中的子句顺序排列，并且没有缺少
                
                doc_text.append(doc_clauses[i]['clause']) #子句的文本内容
                if self.args.iemocaptest == True and self.data_type=='test':
                    doc_emotion_category.append(self.label_vocab['stoi']['neu'])
                else:
                    doc_emotion_category.append(self.label_vocab['stoi']['neutral']) #将子句的情感置为‘neutral’->'0',为啥呀，不应该是将token->int吗
                doc_emotion_token.append(doc_clauses[i]['emotion_token']) #数据集自带的子句的情感token
                doc_str+='<s>'+doc_clauses[i]['clause']+' </s> ' #记录文档中所有子句的text，中间由<s></s>分割
            
            y_emotions_list.append(y_emotions)###情感标签汇总
            y_causes_list.append(y_causes)###原因标签汇总
            doc_text_list.append(doc_text)###文档文本内容汇总
            doc_speaker_list.append(doc_speaker)###文档中说话人汇总
            doc_emotion_category_list.append(doc_emotion_category)###文档emotion_category汇总
            doc_emotion_token_list.append(doc_emotion_token)###emotion_token汇总
            indexed_tokens = self.bert_tokenizer.encode(doc_str.strip(), add_special_tokens=False) #RobertaTokenizer 只返回编码的结果（input_ids），即每个token的idx,eg:[101, 6848, 2885, 4403, 3736,...]
            clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 0] #返回inputs_idx=0的token的位置，即子句token开始的位置
            # 在RobertaTokenizer中，<s></s>是特殊token（等价于bert中的[CLS][SEP]）,<s>的idx为0
            doc_token_len = len(indexed_tokens) #文档总token长度

            segments_ids = []
            segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 0] #子句token开始的位置
            segments_indices.append(len(indexed_tokens)) #将文档token总数添加在所有子句token索引的结尾,为了计算最后一个子句的token数量
            for i in range(len(segments_indices)-1):
                semgent_len = segments_indices[i+1] - segments_indices[i] #子句的token长度
                """
                    为什么要以除以2的方式分割？？数据集一定是A说一句话后B回答一句吗？？？
                """
                if i % 2 == 0: 
                    segments_ids.extend([0] * semgent_len) 
                else:
                    segments_ids.extend([1] * semgent_len)

            assert len(clause_indices) == doc_len
            assert len(segments_ids) == len(indexed_tokens)
            
            bert_token_idx_list.append(indexed_tokens)#文档的所有token_idx
            bert_clause_idx_list.append(clause_indices)#子句token开始的位置
            bert_segments_idx_list.append(segments_ids)#同一个子句中所有token的idx均相同，eg:[0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0]
            bert_token_lens_list.append(doc_token_len)#文档总token长度
            
        return doc_id_list,doc_len_list,doc_couples_list,doc_speaker_list,doc_cls_list, \
        y_emotions_list,y_causes_list,doc_text_list,doc_emotion_category_list,doc_emotion_token_list, \
            bert_token_idx_list,bert_clause_idx_list,bert_segments_idx_list,bert_token_lens_list
    
    #截断
    def token_trunk(self, bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens,
                    doc_couples, y_emotions, y_causes, doc_len,doc_speaker,doc_emotion_category):
        """
            bert_token_idx:文档所有token的idx;
            bert_clause_idx:子句token开始的位置
            bert_segments_idx:同一个子句中所有token的idx均相同，eg:[0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0]
            bert_token_lens:文档总token长度
            doc_couples:文档中的情感原因对
            y_emotions: 文档情感标签汇总，若子句为情感子句置1
            y_causes:文档原因标签汇总，若子句为原因子句置1
            doc_len:文档中子句的个数
            doc_speaker:每个子句的speaker
            doc_emotion_category:子句的情感类别，初始均为“neutral”的索引 即“0”
        """
        # TODO: cannot handle some extreme cases now
        emotion, cause = doc_couples[0]
        if emotion > doc_len / 2 and cause > doc_len / 2:#这个判断是为了？？？ 猜测doc_couples顺序排列的，若情感原因对都出现在后几句话，则前面的话相对于不怎么重要，可以截断
            i = 0
            while True:
                temp_bert_token_idx = bert_token_idx[bert_clause_idx[i]:]
                if len(temp_bert_token_idx) <= 512:                      #这是为了让每个文档的token长度控制在512以内？截断？？类似于nlp的多截少补中的多截？？
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[cls_idx:]
                    bert_segments_idx = bert_segments_idx[cls_idx:]
                    bert_clause_idx = [p - cls_idx for p in bert_clause_idx[i:]]
                    doc_couples = [[emotion - i, cause - i]]
                    y_emotions = y_emotions[i:]
                    y_causes = y_causes[i:]
                    doc_speaker=doc_speaker[i:]
                    doc_emotion_category=doc_emotion_category[i:]
                    doc_len = doc_len - i
                    break
                i = i + 1
        if emotion < doc_len / 2 and cause < doc_len / 2: #若情感原因对都出现在前几句话，则后面的话相对于不怎么重要，可以截断
            i = doc_len - 1
            while True:
                temp_bert_token_idx = bert_token_idx[:bert_clause_idx[i]]
                if len(temp_bert_token_idx) <= 512:
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[:cls_idx]
                    bert_segments_idx = bert_segments_idx[:cls_idx]
                    bert_clause_idx = bert_clause_idx[:i]
                    y_emotions = y_emotions[:i]
                    y_causes = y_causes[:i]
                    doc_speaker=doc_speaker[:i]
                    doc_emotion_category=doc_emotion_category[:i]
                    doc_len = i
                    break
                i = i - 1
        #没有其他判断条件，可能是认为前1/2的话不会导致后1/2的情感，反正同理。 但不能保证确实不存在这种极端的情况，但是忽略
        return bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens, \
               doc_couples, y_emotions, y_causes, doc_len,doc_speaker,doc_emotion_category

    def get_adj(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:#对于每一篇文档的speaker
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                cnt = 0
                for j in range(i - 1, -1, -1):  #将该子句与过去的子句连接起来，但连接的上下文window有限制，并不是全部连接。           
                    a[i,j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt==self.args.windowp:#windowp: context window size for constructing edges in graph model for past utterances
                            break
            adj.append(a)#为每一个文档构建一个邻接矩阵
        return torch.stack(adj)
    
    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot encoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)): #如果想保存节点i的前辈节点的信息，剩下的mask，不应该是 for j in range(0,i+1): 吗？？？ 实际上他是全获取了，在后面用的过程中截断
                    if speaker[i] == speaker[j]:
                        s[i,j] = 1
                        s_onehot[i,j,1] = 1
                    else:
                        s_onehot[i,j,0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)
     
    def pad_matrices(self,doc_len):
        N = max(doc_len)
        adj_b = []
        for dl in doc_len:
            adj = np.ones((dl, dl))
            adj = sp.coo_matrix(adj)
            adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                                shape=(N, N), dtype=np.float32)
            adj_b.append(adj.toarray())
        return adj_b

    def collate_fn(self,batch):
        '''
        :param data:
            doc_id,doc_len,doc_couples,y_emotions,y_causes,doc_speaker,doc_text,doc_cls,
            doc_emotion_category,doc_emotion_token
        :return:
            B:batch_size  N:batch_max_doc_len
            batch_ids:(B)
            batch_doc_len(B)
            batch_pairs(B,) not a tensor
            label_emotions:(B,N) padded
            label_causes:(B,N) padded
            batch_doc_speaker:(B,N) padded
            features: (B, N, D) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            batch_doc_emotion_category: (B, ) not a tensor
            batch_doc_emotion_token:(B, ) not a tensor
            batch_utterances:  not a tensor
            batch_utterances_mask:(B,N) 1表示该子句存在,0表示该子句不存在
            batch_uu_mask (B,N,N)
        '''
        
        doc_id,doc_len,doc_couples,y_emotions,y_causes,doc_speaker,doc_text,doc_cls, doc_emotion_category, \
        doc_emotion_token,bert_token_b, bert_segment_b, bert_clause_b, bert_token_lens_b=zip(*batch)
        
        max_dialog_len=max(int(length) for length in doc_len) #所有文档中对话最多的个数
        #doc_cls中每一个cls是一个二维特征，例如文档由5个子句组成，则该文档的特征shape为[5,dim]
        features=pad_sequence([torch.FloatTensor(cls) for cls in doc_cls],batch_first=True)# 所有文档的初始特征, padding_value: float = 0
        #feature:[batchsize,max_dialog_len,dim]
        """
        pad_sequence:
            Pad a list of variable length Tensors with padding_value
            pad_sequence stacks a list of Tensors along a new dimension, and pads them to equal length. 
            Example:
            >>> from torch.nn.utils.rnn import pad_sequence
            >>> a = torch.ones(25, 300)
            >>> b = torch.ones(22, 300)
            >>> c = torch.ones(15, 300)
            >>> pad_sequence([a, b, c]).size()
            torch.Size([25, 3, 300])
            Note:
                This function returns a Tensor of size T x B x * or B x T x * where T is the length of the longest sequence. 
                This function assumes trailing dimensions and type of all the Tensors in sequences are same.
        """
        #假设一个文档有5个子句，其y_emotions为[0,1,0,1,1],label_emotions的size为[batchsize,max_dialog_len]
        label_emotions=pad_sequence([torch.LongTensor(emo) for emo in y_emotions],batch_first=True,padding_value=-1) #(B,N)包含三种，1，0，-1，其中-1代表该句不存在
        label_causes=pad_sequence([torch.LongTensor(cau) for cau in y_causes],batch_first=True,padding_value=-1)#(B,N)包含三种，1，0，-1，其中-1代表该句不存在
        
        adj=self.get_adj([sp for sp in doc_speaker],max_dialog_len) #为每一个文档构建一个邻接矩阵，torch.stack(dim=0)
        s_mask,s_mask_onehot=self.get_s_mask([sp for sp in doc_speaker],max_dialog_len) 
        #s_mask: (B, N, N) s_mask[:,i,:] 表示节点i的前辈的说话人信息，其中1表示同一说话人，0表示不同的说话人
        #s_mask_onehot (B, N, N, 2) onehot encoding of s_mask

        batch_doc_speaker=pad_sequence([torch.LongTensor(speaker) for speaker in doc_speaker], batch_first=True,padding_value=-1)
        batch_ids=[docid for docid in doc_id]
        batch_doc_len=torch.LongTensor([doclen for doclen in doc_len])
        batch_pairs=[pairs for pairs in doc_couples]
        
        # batch_doc_emotion_category=[dec for dec in doc_emotion_category]
        batch_doc_emotion_category=pad_sequence([torch.LongTensor(dec) for dec in doc_emotion_category], batch_first=True,padding_value=-1)
        batch_doc_emotion_token=[det for det in doc_emotion_token] #是token，不是数
        
        batch_utterances=[utt for utt in doc_text]#中间由<s></s>分割的字符串
        zero=torch.zeros_like(label_emotions)
        batch_utterances_mask=torch.ones_like(label_emotions)
        batch_utterances_mask=torch.where(label_emotions==-1,zero,batch_utterances_mask)
        
        batch_uu_mask=self.pad_matrices(doc_len) 
        #padding,由于前面的padding操作，根据的是所有文档的最大话语数，所以有些文档被加入了不存在的子句，需要将这些子句的邻接矩阵padding为0，存在的子句为1
        
        bert_token_b = pad_sequence(bert_token_b, batch_first=True, padding_value=0)
        #bert_token_b: [num_doc(batchsize),max_num_token]  max_num_token:该batch中的最大token数
        bert_segment_b = pad_sequence(bert_segment_b, batch_first=True, padding_value=0)
        bert_clause_b = pad_sequence(bert_clause_b, batch_first=True, padding_value=0)
        
        bsz, max_len = bert_token_b.size()
        bert_masks_b = np.zeros([bsz, max_len], dtype=np.float)
        for index, seq_len in enumerate(bert_token_lens_b):#bert_token_lens_b:每篇文档的token数量，list
            bert_masks_b[index][:seq_len] = 1
        bert_masks_b = torch.FloatTensor(bert_masks_b)
        
        assert bert_segment_b.shape == bert_token_b.shape
        assert bert_segment_b.shape == bert_masks_b.shape
        
        return batch_ids,batch_doc_len,batch_pairs,np.array(label_emotions),np.array(label_causes), \
            batch_doc_speaker,features,adj, \
            s_mask,s_mask_onehot,batch_doc_emotion_category,batch_doc_emotion_token,batch_utterances, \
            np.array(batch_utterances_mask),np.array(batch_uu_mask),bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b
        
                       
                