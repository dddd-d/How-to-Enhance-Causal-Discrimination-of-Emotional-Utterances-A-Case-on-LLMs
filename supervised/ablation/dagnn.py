import torch
import torch.nn as nn
import torch .nn.functional as F
from ablation.gnn_utils import *


class DAGNN(nn.Module):
    def __init__(self,args):
        super(DAGNN,self).__init__()
        self.args=args
        self.dropout=nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(args.emb_dim, args.gnn_hidden_dim)
        self.gnn_layers=args.gnn_layers
        gats=[]
        for _ in range(args.gnn_layers): #3
            gats += [GAT_dagnn(args.gnn_hidden_dim)]#args.gnn_hidden_dim=300
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
        
        self.nodal_att_type = args.nodal_att_type #('--nodal_att_type', type=str, default=None, choices=['global','past'], help='type of nodal attention')
        in_dim = args.gnn_hidden_dim * (args.gnn_layers + 1) + args.emb_dim #args.emb_dim=768
        self.attentive_node_features = attentive_node_features(in_dim)
        
        # output mlp layers
        layers = [nn.Linear(in_dim, args.gnn_hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.gnn_hidden_dim, args.gnn_hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.gnn_hidden_dim, args.feat_dim)] #args.feat_dim=768

        self.out_mlp = nn.Sequential(*layers)
        
    def forward(self,features,adj,s_mask,s_mask_onehot,lengths):
        '''
        :param features: (B, N, D)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :param s_mask_onehot: (B, N, N, 2)
        :return:
        '''
        num_utter = features.size()[1] #padding后每篇文档的子句数量
        H0 = F.relu(self.fc1(features))#(B,N,D) to (B,N,H) B存在8，4，7等不同，N不同，D1024，H300 输入特征维度-linear->gnn维度
        # H0 = self.dropout(H0)
        H = [H0]### H是一个长度为1的list
        adjB=torch.zeros([features.size()[0],1,num_utter]).cuda()
        adjB_list=[adjB]
        for l in range(self.args.gnn_layers):
            C = self.grus_c[l](H[l][:,0,:]).unsqueeze(1) #H[0][:,0,:]表示每个样本（共batch个）的第一句话:size（B，H），H[1]是后面加入的第一层图卷积，
            #先采用第0（l=0）层的GRUcell学习，再升维：(B,H)->(B,1,H)
            M = torch.zeros_like(C).squeeze(1) # M（B，H） 
            # P = M.unsqueeze(1) 
            P = self.grus_p[l](M, H[l][:,0,:]).unsqueeze(1)  
            H1 = C+P  
            for i in range(1, num_utter):#先计算H[:,1,:]基于H[:,0,:]的注意力，然后再计算H[:,2,:]基于H[:,0+1,:]的注意力，以此类推
                # print(i,num_utter)
                
                B, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask[:,i,:i]) #三层GAT GAT_dagnn
                B=nn.ZeroPad2d(padding=(0,num_utter-B.size()[-1],0,0))(B.squeeze(1)).unsqueeze(1)
                C = self.grus_c[l](H[l][:,i,:], M).unsqueeze(1)
                P = self.grus_p[l](M, H[l][:,i,:]).unsqueeze(1)   
                
                H_temp = C+P
                H1 = torch.cat((H1 , H_temp), dim = 1)  
                if i==0:
                    
                    adjB=B
                else:
                    adjB=torch.cat((adjB,B),dim=1)
                # print('H1', H1.size())
                # print('----------------------------------------------------')
            H.append(H1)#D=1200
            adjB_list.append(adjB)
        H.append(features)#D=1024
        H = torch.cat(H, dim = 2) #(B,N,D) D=2224

        H = self.attentive_node_features(H,lengths,self.nodal_att_type) #(B,N,D) D=2224
        
        logits = self.out_mlp(H)
        
        return logits,adjB_list[2]