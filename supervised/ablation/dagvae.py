from re import X
from turtle import forward
import torch
import torch.nn as nn
import torch .nn.functional as F
from ablation.gnn_utils import *

class CAVAE_dagnn(nn.Module):
    def __init__(self,args):
        super(CAVAE_dagnn,self).__init__()
        self.args=args
        self.encoder=DAGNNencoder(args)
        self.decoder=DAGNNdecoder(args)
        
    
    def forward(self,features,adj,s_mask,s_mask_onehot,lengths):
        fU,adjB_list=self.encoder(features,adj,s_mask,s_mask_onehot,lengths) #fU (B,N,code_dim) adjB_list:[(B,1,num_utter),(B,num_utter,num_utter),(B,num_utter,num_utter),(B,num_utter,num_utter)]
        X,b_inv=self.decoder(fU,adj,s_mask,s_mask_onehot,lengths,adjB_list)

        return X,b_inv
        
class DAGNNencoder(nn.Module):
    def __init__(self,args):
        super(DAGNNencoder,self).__init__()
        self.args=args
        self.dropout=nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(args.emb_dim, args.gnn_hidden_dim) #args.gnn_hidden_dim=300
        self.gnn_layers=args.gnn_layers
        gats=[]
        for _ in range(args.gnn_layers):
            gats += [GAT_encoder(args.gnn_hidden_dim)]
        self.gather = nn.ModuleList(gats)
        
        grus_c = []
        for _ in range(args.gnn_layers):
            grus_c += [nn.GRUCell(args.gnn_hidden_dim, args.gnn_hidden_dim)]
        self.grus_c = nn.ModuleList(grus_c)#两层 GRUcell

        grus_p = []
        for _ in range(args.gnn_layers):
            grus_p += [nn.GRUCell(args.gnn_hidden_dim, args.gnn_hidden_dim)]
        self.grus_p = nn.ModuleList(grus_p)#两层 GRUcell
        
        fcs = []
        for _ in range(args.gnn_layers):
            fcs += [nn.Linear(args.gnn_hidden_dim * 2, args.gnn_hidden_dim)]
        self.fcs = nn.ModuleList(fcs) #没用到
    
        #这个in_dim没用，而且nodal_att_type必须为None，若不为None，下面代码注释掉的H=H.append(feature)要放出来，不然维度对不上
        self.nodal_att_type = args.nodal_att_type #args:('nodal_att_type', default=None, choices=['global','past'], help='type of nodal attention')
        in_dim = args.gnn_hidden_dim * (args.gnn_layers + 1) + args.emb_dim #args.emb_dim=1024?
        self.attentive_node_features = attentive_node_features(in_dim)
        
        # output mlp layers
        layers = [nn.Linear(args.gnn_hidden_dim * (args.gnn_layers + 1), args.gnn_hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.gnn_hidden_dim, args.gnn_hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.gnn_hidden_dim, args.code_dim)] #args.code_dim=192

        self.out_mlp = nn.Sequential(*layers)
        self.largefc=nn.Linear(self.args.code_dim,args.emb_dim)
             
    def forward(self,features,adj,s_mask,s_mask_onehot,lengths):
        '''
        :param features: (B, N, D)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :param s_mask_onehot: (B, N, N, 2)
        :return:
        '''
        # if features.size()[-1]==self.args.code_dim:
        #     input=F.relu(self.largefc(features))
        # else:
        #     input=features
        num_utter = features.size()[1] #padding后每篇文档的子句数量
        H0 = F.relu(self.fc1(features))#(B,N,D) to (B,N,H) B存在8，4，7等不同，N不同，D1024，H300 输入特征维度-linear->gnn维度
        # H0 = self.dropout(H0)
        H = [H0]### H是一个长度为1的list
        adjB=torch.zeros([features.size()[0],1,num_utter]).cuda()
        adjB_list=[adjB]
        for l in range(self.args.gnn_layers):
            C = self.grus_c[l](H[l][:,0,:]).unsqueeze(1) #H[0][:,0,:]表示每个样本（共batch个）的第一句话size:（B，H），H[1]是后面加入的第一层图卷积，
            #先采用第0（l=0）层的GRUcell学习，再升维：(B,H)->(B,1,H)
            M = torch.zeros_like(C).squeeze(1) # M（B，H） 
            # P = M.unsqueeze(1) 
            P = self.grus_p[l](M, H[l][:,0,:]).unsqueeze(1)  
            H1 = C+P #C and P (B,1,H)
            adjB=torch.zeros([features.size()[0],1,num_utter]).cuda()
            for i in range(1, num_utter):#先计算H[:,1,:]基于H[:,0,:]的注意力，然后再计算H[:,2,:]基于H[:,0+1,:]的注意力，以此类推
                # print(i,num_utter)
                #Q是当前句，K和V（H1）都是之前句的集合，adj和s_mask也是只有之前句子的集合，所以到做到I-B得在gnn具体计算时操作
                B, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask[:,i,:i])  #三层GAT GAT_dagnn
                #GAT_dagnn : Q, K, V, adj, s_mask
                #GAT_dagnn ：return attn_weight（Q*K）(B,1,n) 此时N≠n, attn_sum（注意力加权后的特征输出）(B,H)
                
                B=nn.ZeroPad2d(padding=(0,num_utter-B.size()[-1],0,0))(B.squeeze(1)).unsqueeze(1)
                """
                    n.ZeroPad2d(),指定tensor的四个方向上的填充，比如左边添加1dim、右边添加2dim、上边添加3dim、下边添加4dim，即指定paddin参数为（1，2，3，4）
                    pad = nn.ZeroPad2d(padding=(0,1,0,0))
                    B = pad(B)
                """
                #B(B,1,N)->(B,N)-在右侧padding恢复原长->(B,num_utter)->(B,1,num_utter),N是当前句之前的子句的数量
                
                C = self.grus_c[l](H[l][:,i,:], M).unsqueeze(1)
                P = self.grus_p[l](M, H[l][:,i,:]).unsqueeze(1)   
                
                H_temp = C+P
                H1 = torch.cat((H1 , H_temp), dim = 1)  #H1 i循环一次H1就从（B，n，H）变为（B，n+1，H）
                if i==0:  
                    adjB=B
                else:
                    adjB=torch.cat((adjB,B),dim=1)
                # print('H1', H1.size())
                # print('----------------------------------------------------')
            H.append(H1)# H=[tensor[H],tensor[H1],……] #记录三层GNN出来的特征
            adjB_list.append(adjB) #加权邻接矩阵
        #H.append(features)#D=1024
        H = torch.cat(H, dim = 2) #(B,N,H') H'=300+300*3+1024
        #实际上并没有用到这个函数，直接返回的H。这个函数接受的特征的size 为 300+300*3+1024，需要将feature加入H（即前面注释掉的部分）
        #但由于不经过这个函数，我们获得的H的 size 为 300+300*3
        H = self.attentive_node_features(H,lengths,self.nodal_att_type) #(B,N,H') 
        
        logits = self.out_mlp(H) #H'->H -> code_dim =192
        
        
        return logits,adjB_list

class DAGNNdecoder(nn.Module):
    def __init__(self,args):
        super(DAGNNdecoder,self).__init__()
        self.args=args
        self.dropout=nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(args.code_dim, args.gnn_hidden_dim)
        self.gnn_layers=args.gnn_layers
        gats=[]
        for _ in range(args.gnn_layers):
            gats += [GNN_decoder(args.gnn_hidden_dim)]
        self.gather = nn.ModuleList(gats)
        
        grus_c = []
        for _ in range(args.gnn_layers):
            grus_c += [nn.GRUCell(args.gnn_hidden_dim, args.gnn_hidden_dim)]
        self.grus_c = nn.ModuleList(grus_c)#两层 GRUcell

        grus_p = []
        for _ in range(args.gnn_layers):
            grus_p += [nn.GRUCell(args.gnn_hidden_dim, args.gnn_hidden_dim)]
        self.grus_p = nn.ModuleList(grus_p)#两层 GRUcell
        
        fcs = []
        for _ in range(args.gnn_layers):
            fcs += [nn.Linear(args.gnn_hidden_dim * 2, args.gnn_hidden_dim)]
        self.fcs = nn.ModuleList(fcs)
        
        self.nodal_att_type = args.nodal_att_type
        in_dim = args.gnn_hidden_dim * (args.gnn_layers + 1) + args.emb_dim
        self.attentive_node_features = attentive_node_features(in_dim)
        
        # output mlp layers
        layers = [nn.Linear(args.gnn_hidden_dim * (args.gnn_layers + 1), args.gnn_hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.gnn_hidden_dim, args.gnn_hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.gnn_hidden_dim, args.feat_dim)]

        self.out_mlp = nn.Sequential(*layers)
        
    def forward(self,features,adj,s_mask,s_mask_onehot,lengths,adjB_list):
        '''
        :param features: (B, N, code_dim)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :param s_mask_onehot: (B, N, N, 2)
        :param lengths:(B,)
        :param adjB_list:[(B,1,N),(B,N,N),(B,N,N),(B,N,N)]
        '''
        
        num_utter = features.size()[1]
        I=torch.eye(num_utter).repeat(features.size()[0],1,1).cuda()# I:(B,N,N)
        """
            torch.eye:生成对角线全1，其余部分全0的tensor
            torch.repeat()某一维度扩张
        """
        H0 = F.relu(self.fc1(features))#(B,N,code_dim) to (B,N,H) B存在8，4，7等不同，N不同，H300
        # H0 = self.dropout(H0)
        H = [H0]### H是一个长度为1的list
        for l in range(self.args.gnn_layers):
            C = self.grus_c[l](H[l][:,0,:]).unsqueeze(1) #H[0][:,0,:]表示每个样本（共batch个）的第一句话，（B，H），H[1]是后面加入的第一层图卷积，
            #先变成（B，1，H），然后采用第0（l=0）层的GRUcell学习
            M = torch.zeros_like(C).squeeze(1) # M（B，H） 
            # P = M.unsqueeze(1) 
            P = self.grus_p[l](M, H[l][:,0,:]).unsqueeze(1)  
            H1 = C+P#C and P (B,1,H)
            #求 I-B 的逆
            b_inv= torch.linalg.solve(I, (I-adjB_list[l+1])) #adjB_list[0]没有用
            
            for i in range(1, num_utter):#先计算H[:,1,:]基于H[:,0,:]的注意力，然后再计算H[:,2,:]基于H[:,0+1,:]的注意力，以此类推
                #Q是当前句，K和V（H1）都是之前句的集合，adj和s_mask也是只有之前句子的集合，所以到做到I-B得在gnn具体计算时操作

                #注意这里用的不再是GAT而是GNN
                M = self.gather[l](H[l][:,i,:], H1, H1, b_inv[:,i,:i], s_mask[:,i,:i])
                
                C = self.grus_c[l](H[l][:,i,:], M).unsqueeze(1)
                P = self.grus_p[l](M, H[l][:,i,:]).unsqueeze(1)   
                
                H_temp = C+P
                if i==0:
                    H1=H_temp
                else:
                    H1 = torch.cat((H1 , H_temp), dim = 1)  #H1 i循环一次H1就从（B，n，D）变为（B，n+1，D）
                    
                # print('H1', H1.size())
                # print('----------------------------------------------------')
            H.append(H1)# H=[tensor[H],tensor[H1],……]
            
        # H.append(features)#D=1024
        H = torch.cat(H, dim = 2) 

        H = self.attentive_node_features(H,lengths,self.nodal_att_type) 
        
        logits = self.out_mlp(H)
        feature_map= torch.linalg.solve(I, (I-adjB_list[2])) #没有用
        
        return logits,feature_map


