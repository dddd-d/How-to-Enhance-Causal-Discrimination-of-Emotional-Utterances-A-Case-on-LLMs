
from sympy import false
import torch
from utils import *
from plot import *
import matplotlib.pyplot as plt
import wandb
def train_eval(model,dataloader,fold,epoch,args,optimizer,scheduler,logger,writer,train=False):
    assert not model or dataloader or optimizer or scheduler!= None
    if train:
        model.train()
        logger.info('########################Training######################')
        # dataloader = tqdm(dataloader)
    else:
        model.eval()
        logger.info('########################Evaling######################')
        
    ####统计的数据#####
    doc_id_all,doc_couples_all,doc_couples_pred_all=[],[],[]   
    y_causes_b_all = []
    
     
    for train_step, batch in enumerate(dataloader, 1):
        batch_ids,batch_doc_len,batch_pairs,label_emotions,label_causes,batch_doc_speaker,features,adj,s_mask, \
            s_mask_onehot,batch_doc_emotion_category,batch_doc_emotion_token,batch_utterances,batch_utterances_mask,batch_uu_mask, \
                bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b=batch
        
        features = features.cuda()
        adj = adj.cuda()
        s_mask = s_mask.cuda()
        s_mask_onehot = s_mask_onehot.cuda()
        batch_doc_len = batch_doc_len.cuda()
        batch_doc_emotion_category=batch_doc_emotion_category.cuda()
        
        couples_pred, emo_cau_pos, pred1_e, pred1_c,pred2_e, pred2_c,adj_map = model(features,adj,s_mask,s_mask_onehot,batch_doc_len,batch_uu_mask, \
            bert_token_b,bert_masks_b,bert_clause_b)
        # if epoch==50: 
        #     for i in range(len(batch_doc_len)):
        #         if batch_doc_len[i]==7 or batch_doc_len[i]==8:
        #             img,im=plot_cka_matrix(IBr[i],batch_doc_len[i])
        #             texts = annotate_heatmap(im, valfmt="{x:.2f}")
        #             img.savefig('savefig/{}+{}+{}+cgnn.jpg'.format(batch_doc_len[i],batch_ids[i],str(batch_pairs[i])))
        #     plt.show()
        
        
         
         
        loss_e, loss_c = model.loss_pre(pred1_e, pred1_c,pred2_e, pred2_c, label_emotions, label_causes, batch_utterances_mask)
        loss_couple, doc_couples_pred = model.loss_rank(couples_pred, emo_cau_pos, batch_pairs, batch_utterances_mask)
        # loss_KL=model.loss_vae(batch_uu_mask,adj_map1,adj_map2)
        #loss_KL=0
        if args.withvae==True:
            loss_KL=model.loss_vae(pred1_e, pred1_c,pred2_e, pred2_c)
        else:
            loss_KL=0
        #loss_KL=1
        if len(dataloader)==47:
            logger.info('VALID# fold: {}, epoch: {}, iter: {},  loss_e: {},  loss_c: {},  loss_couple:{},   loss_KL:{}'. \
                                format(fold,   epoch,    train_step, loss_e,     loss_c,     loss_couple,loss_KL))
            writer.add_scalars('valid_loss',{'loss_e':loss_e,'loss_c':loss_c,'loss_couple':loss_couple, \
                                              'loss_KL':loss_KL},train_step+len(dataloader)*epoch)
        if len(dataloader)==257:
            logger.info('TEST# fold: {}, epoch: {}, iter: {},  loss_e: {},  loss_c: {},  loss_couple:{},   loss_KL:{}'. \
                                format(fold,   epoch,    train_step, loss_e,     loss_c,     loss_couple,loss_KL))
            writer.add_scalars('test_loss',{'loss_e':loss_e,'loss_c':loss_c,'loss_couple':loss_couple, \
                                             'loss_KL':loss_KL},train_step+len(dataloader)*epoch)
        
        
        #如果不是训练集只计算这两个loss，但是不bp
        if train:
            logger.info('TRAIN# fold: {}, epoch: {}, iter: {},  loss_e: {},  loss_c: {},  loss_couple:{},   loss_KL:{}'. \
                                format(fold,   epoch,    train_step, loss_e,     loss_c,     loss_couple,loss_KL))
            writer.add_scalars('train_loss',{'loss_e':loss_e,'loss_c':loss_c,'loss_couple':loss_couple, \
                                                 'loss_KL':loss_KL},train_step+len(dataloader)*epoch)
            loss = loss_e + loss_c+loss_couple+loss_KL
            wandb.log({'epoch': epoch,  'step':train_step+len(dataloader)*epoch,'loss_all':loss,'loss_couple':loss_couple,'loss_e':loss_e,'loss_c':loss_c,'loss_KL':loss_KL})
            loss = loss / args.gradient_accumulation_steps
            loss.backward()#计算梯度
            if train_step % args.gradient_accumulation_steps == 0:
                optimizer.step()#反向传播，两个batch传播一次，分别累计loss
                scheduler.step()
                model.zero_grad()
                
        doc_id_all.extend(batch_ids)
        doc_couples_all.extend(batch_pairs)
        doc_couples_pred_all.extend(doc_couples_pred)
        y_causes_b_all.extend(list(label_causes))
    if train==False:
    #####若为test或者valid计算指标######
        doc_couples_pred_all = lexicon_based_extraction(doc_id_all, doc_couples_pred_all,fold=fold)
        metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c = eval_func(doc_couples_all, \
            doc_couples_pred_all, y_causes_b_all)
        return metric_ec_p, metric_ec_n, metric_ec_avg, metric_e, metric_c, doc_id_all, doc_couples_all, doc_couples_pred_all
        
def lexicon_based_extraction(doc_ids, couples_pred,fold):
    emotional_clauses = read_b('data/dailydialog/fold%s/sentimental_clauses.pkl'%(fold))#每个对话情感标签的顺序表

    couples_pred_filtered = []
    for i, (doc_id, couples_pred_i) in enumerate(zip(doc_ids, couples_pred)):
        top1, top1_prob = couples_pred_i[0][0], couples_pred_i[0][1]
        couples_pred_i_filtered = [top1]

        emotional_clauses_i = emotional_clauses[doc_id]
        for couple in couples_pred_i[1:]:
            if couple[0][0] in emotional_clauses_i and logistic(couple[1]) > 0.5 and couple[0][0]>=couple[0][1]:
                couples_pred_i_filtered.append(couple[0])

        couples_pred_filtered.append(couples_pred_i_filtered)
    return couples_pred_filtered