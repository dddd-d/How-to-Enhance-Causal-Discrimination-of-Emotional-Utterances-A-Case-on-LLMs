import torch
import torch.nn as nn
import torch .nn.functional as F
import numpy as np


class RankNN(nn.Module):
    def __init__(self, args):
        super(RankNN, self).__init__()
        self.K = args.rank_K #default=12
        self.pos_emb_dim = args.rank_pos_emb_dim #default=50
        self.pos_layer = nn.Embedding(2*self.K + 1, self.pos_emb_dim) #这个位置向量的个数未必是2*self.K+1吧可以写num_utter吧，这样虽然多了一些位置但是能用
        nn.init.xavier_uniform_(self.pos_layer.weight)

        #self.feat_dim = int(args.gnn_hidden_dim * (args.gnn_layers + 1) + args.emb_dim)
        self.feat_dim=args.feat_dim
        self.rank_feat_dim = 2*self.feat_dim + self.pos_emb_dim
        self.rank_layer1 = nn.Linear(self.rank_feat_dim, self.rank_feat_dim)
        self.rank_layer2 = nn.Linear(self.rank_feat_dim, 1)

    def forward(self, doc_sents_h):
        batch, _, _ = doc_sents_h.size() #求得batch 
        couples, rel_pos, emo_cau_pos = self.couple_generator(doc_sents_h, self.K)#K 是考虑附近多少个子句的限制
        #couple_generator
        #couples:(B,mask_N*N,2*feat_dim) P:原先是N*N mask之后只保留了True值，维度发生变化
        #rel_pos是一个序列，里面是考虑的子句位置，(B,masked_N*N)
        #emo_cau_pos是考虑的子句对 list len:masked_N*N [emo_pos,cau_pos]
        
        rel_pos = rel_pos + self.K
        #eg: k=2, N=5
        #rel_pos: tensor([[ 0,  1,  2, -1,  0,  1,  2, -2, -1,  0,  1,  2, -2, -1,  0,  1, -2, -1, 0],[...],...])  (B,masked_N*N)
        #rel_pos + self.K: tensor([[2,  3,  4,  1,  2,  3,  4,  0,  1,  2,  3,  4, 0,  1,  2,  3,  0,  1, 2],[...],...]) (B,masked_N*N) #masked的真实位置pos
        rel_pos_emb = self.pos_layer(rel_pos)#(B,masked_N*N,emb_dim) 位置信息
        kernel = self.kernel_generator(rel_pos)#(masked_N*N,masked_N*N)
        kernel = kernel.unsqueeze(0).expand(batch, -1, -1)#(B,masked_N*N,masked_N*N)
        rel_pos_emb = torch.matmul(kernel, rel_pos_emb)#(B,masked_N*N,emb_dim)
        couples = torch.cat([couples, rel_pos_emb], dim=2)#(B,masked_N*N,2*feat_dim+emb_dim) 

        couples = F.relu(self.rank_layer1(couples))
        couples_pred = self.rank_layer2(couples)# （B，masked_N*N,1）
        return couples_pred.squeeze(2), emo_cau_pos

    def couple_generator(self, H, k):
        batch, seq_len, feat_dim = H.size()
        
        P_left = torch.cat([H] * seq_len, dim=2)#[H] * seq_len 为复制20个H，放入一个list中，然后在维度上全部拼接
        P_left = P_left.reshape(-1, seq_len * seq_len, feat_dim)# （B，N*N，D）
        P_right = torch.cat([H] * seq_len, dim=1)#（B，N*N，D）
        P = torch.cat([P_left, P_right], dim=2) #这个P （B，N*N，2D），包含了任意两个子句的组合

        base_idx = np.arange(1, seq_len + 1)#从1开始的字句长度序列
        emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]#N*N的序列，N个1，N个2……N个N
        cau_pos = np.concatenate([base_idx] * seq_len, axis=0)#N*N的序列，1-N，1-N，……
        """
            eg:seq_len = 5
            emo_pos = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5]
            cau_pos = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
        """

        rel_pos = cau_pos - emo_pos
        rel_pos = torch.LongTensor(rel_pos).cuda()#N*N，0到N-1 接 -1到N-2
        emo_pos = torch.LongTensor(emo_pos).cuda()
        cau_pos = torch.LongTensor(cau_pos).cuda()

        if seq_len > k + 1:
            rel_mask = np.array(list(map(lambda x: -k <= x <= k, rel_pos.tolist())), dtype=np.int)
            rel_mask = torch.BoolTensor(rel_mask).cuda()#0，1的序列，每个句子前后K个标1，
            """
            eg:k=2
            rel_mask:
            tensor  ([  True,  True,  True, False, False,  
                        True,  True,  True,  True, False,
                        True,  True,  True,  True,  True, 
                        False,  True,  True,  True,  True,
                        False, False,  True,  True,  True
                    ])
            """
            rel_pos = rel_pos.masked_select(rel_mask)
            emo_pos = emo_pos.masked_select(rel_mask)
            cau_pos = cau_pos.masked_select(rel_mask)
            ###都选择mask中为1的值，不再是N*N了，要小一些
            """
            emo_pos:
                tensor([1, 1, 1, 
                        2, 2, 2, 2, 
                        3, 3, 3, 3, 3, 
                            4, 4, 4, 4, 
                                5, 5, 5])
            cau_pos:
                tensor([1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5])
            rel_mask:
                tensor([ 0,  1,  2, -1,  0,  1,  2, -2, -1,  0,  1,  2, -2, -1,  0,  1, -2, -1, 0])
            """
            rel_mask = rel_mask.unsqueeze(1).expand(-1, 2 * feat_dim) # (N*N, 2*feat_dim)
            rel_mask = rel_mask.unsqueeze(0).expand(batch, -1, -1)#(B, N*N, 2*feat_dim)
            P = P.masked_select(rel_mask) #P:(B, N*N, 2*feat_dim) masked_select只保留了没有被mask的值，此时P的第二维不再是N*N
            P = P.reshape(batch, -1, 2 * feat_dim)#保留了rel_mask中为真的值，第二维不是N*N了  P:(B,P,2H)
        assert rel_pos.size(0) == P.size(1)#确认一下两个不是N*N的维度是不是一样的
        rel_pos = rel_pos.unsqueeze(0).expand(batch, -1) #(P,)->(1,P)->(B,P)

        emo_cau_pos = []
        for emo, cau in zip(emo_pos.tolist(), cau_pos.tolist()):
            emo_cau_pos.append([emo, cau]) #所有的伪"情感因果"对
        return P, rel_pos, emo_cau_pos

    def kernel_generator(self, rel_pos):
        n_couple = rel_pos.size(1)#masked_N*N
        rel_pos_ = rel_pos[0].type(torch.FloatTensor).cuda()
        kernel_left = torch.cat([rel_pos_.reshape(-1, 1)] * n_couple, dim=1)
        kernel = kernel_left - kernel_left.transpose(0, 1)
        return torch.exp(-(torch.pow(kernel, 2)))