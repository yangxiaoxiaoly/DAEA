import torch
import torch.nn as nn
import torch.nn.functional as F
from Param import *
import numpy as np
import time
import pickle
from eval_function import cos_sim_mat_generate,batch_topk,hit_res
from mmd import *


def entlist2emb(Model,entids,entid2data,cuda_num):
    """
    return basic bert unit output embedding of entities
    """
    batch_token_ids = []
    batch_mask_ids = []
    for eid in entids:
        temp_token_ids = entid2data[eid][0]
        temp_mask_ids = entid2data[eid][1]

        batch_token_ids.append(temp_token_ids)
        batch_mask_ids.append(temp_mask_ids)

    batch_token_ids = torch.LongTensor(batch_token_ids).cuda(cuda_num)
    batch_mask_ids = torch.FloatTensor(batch_mask_ids).cuda(cuda_num)

    batch_emb = Model(batch_token_ids,batch_mask_ids)
    del batch_token_ids
    del batch_mask_ids
    return batch_emb


def generate_candidate_dict(Model,train_ent1s,train_ent2s,for_candidate_ent1s,for_candidate_ent2s,
                                entid2data,index2entity,
                                nearest_sample_num = NEAREST_SAMPLE_NUM, batch_size = CANDIDATE_GENERATOR_BATCH_SIZE):
    start_time = time.time()
    Model.eval()
    torch.cuda.empty_cache()
    candidate_dict = dict()
    with torch.no_grad():
        #langauge1 (KG1)
        train_emb1 = []
        for_candidate_emb1 = []
        for i in range(0,len(train_ent1s),batch_size):
            temp_emb = entlist2emb(Model,train_ent1s[i:i+batch_size],entid2data,CUDA_NUM).cpu().tolist()
            train_emb1.extend(temp_emb)
        for i in range(0,len(for_candidate_ent2s),batch_size):
            temp_emb = entlist2emb(Model,for_candidate_ent2s[i:i+batch_size],entid2data,CUDA_NUM).cpu().tolist()
            for_candidate_emb1.extend(temp_emb)

        #language2 (KG2)
        train_emb2 = []
        for_candidate_emb2 = []
        for i in range(0,len(train_ent2s),batch_size):
            temp_emb = entlist2emb(Model,train_ent2s[i:i+batch_size],entid2data,CUDA_NUM).cpu().tolist()
            train_emb2.extend(temp_emb)
        for i in range(0,len(for_candidate_ent1s),batch_size):
            temp_emb = entlist2emb(Model,for_candidate_ent1s[i:i+batch_size],entid2data,CUDA_NUM).cpu().tolist()
            for_candidate_emb2.extend(temp_emb)
        torch.cuda.empty_cache()

        #cos sim
        cos_sim_mat1 = cos_sim_mat_generate(train_emb1,for_candidate_emb1)
        cos_sim_mat2 = cos_sim_mat_generate(train_emb2,for_candidate_emb2)
        torch.cuda.empty_cache()
        #topk index
        _,topk_index_1 = batch_topk(cos_sim_mat1,topn=nearest_sample_num,largest=True)
        topk_index_1 = topk_index_1.tolist()
        _,topk_index_2 = batch_topk(cos_sim_mat2,topn=nearest_sample_num,largest=True)
        topk_index_2 = topk_index_2.tolist()
        #get candidate
        for x in range(len(topk_index_1)):
            e = train_ent1s[x]
            candidate_dict[e] = []
            for y in topk_index_1[x]:
                c = for_candidate_ent2s[y]
                candidate_dict[e].append(c)
        for x in range(len(topk_index_2)):
            e = train_ent2s[x]
            candidate_dict[e] = []
            for y in topk_index_2[x]:
                c = for_candidate_ent1s[y]
                candidate_dict[e].append(c)

        #show
        # def rstr(string):
        #     return string.split(r'/resource/')[-1]
        # for e in train_ent1s[100:105]:
        #     print(rstr(index2entity[e]),"---",[rstr(index2entity[eid]) for eid in candidate_dict[e][:6]])
        # for e in train_ent2s[100:105]:
        #     print(rstr(index2entity[e]),"---",[rstr(index2entity[eid]) for eid in candidate_dict[e][:6]])
    print("get candidate using time: {:.3f}".format(time.time()-start_time))
    torch.cuda.empty_cache()
    return candidate_dict




#, Train_gene_target,train_ill_target,test_ill_target,entid2data_target

def train(Model,Criterion,Optimizer,Train_gene,train_ill,test_ill,entid2data, Train_gene_target,train_ill_target,test_ill_target,entid2data_target):
    print("start training...")
    for epoch in range(EPOCH_NUM):
        print("+++++++++++")
        print("Epoch: ",epoch)
        print("+++++++++++")
        
        #generate candidate_dict
        #(candidate_dict is used to generate negative example for train_ILL)
        train_ent1s = [e1 for e1,e2 in train_ill]
        train_ent2s = [e2 for e1,e2 in train_ill]
        for_candidate_ent1s = Train_gene.ent_ids1
        for_candidate_ent2s = Train_gene.ent_ids2
        print("train ent1s num: {} train ent2s num: {} for_Candidate_ent1s num: {} for_candidate_ent2s num: {}"
              .format(len(train_ent1s),len(train_ent2s),len(for_candidate_ent1s),len(for_candidate_ent2s)))
        candidate_dict = generate_candidate_dict(Model,train_ent1s,train_ent2s,for_candidate_ent1s,
                                                     for_candidate_ent2s,entid2data,Train_gene.index2entity)
        Train_gene.train_index_gene(candidate_dict) #generate training data with candidate_dict

        #train
        # epoch_loss,epoch_train_time = ent_align_train(Model,Criterion,Optimizer,Train_gene,entid2data)
        
        #target 
        print("tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt")
        #generate candidate_dict
        #(candidate_dict is used to generate negative example for train_ILL)
        train_ent1s_target = [e1 for e1,e2 in train_ill_target]
        train_ent2s_target = [e2 for e1,e2 in train_ill_target]
        for_candidate_ent1s_target = Train_gene_target.ent_ids1
        for_candidate_ent2s_target = Train_gene_target.ent_ids2
        print("train ent1s num: {} train ent2s num: {} for_Candidate_ent1s num: {} for_candidate_ent2s num: {}"
              .format(len(train_ent1s_target),len(train_ent2s_target),len(for_candidate_ent1s_target),len(for_candidate_ent2s_target)))
        candidate_dict_target = generate_candidate_dict(Model,train_ent1s_target,train_ent2s_target,for_candidate_ent1s_target,
                                                     for_candidate_ent2s_target,entid2data_target,Train_gene_target.index2entity)
        Train_gene_target.train_index_gene(candidate_dict_target) #generate training data with candidate_dict
        
        #train
        # epoch_loss_target,epoch_train_time_target = ent_align_train(Model,Criterion,Optimizer,Train_gene_target,entid2data_target)
        
        epoch_loss,epoch_train_time = ent_align_train(Model,Criterion,Optimizer,Train_gene,entid2data,Train_gene_target,entid2data_target)

        
        
        print("tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt")
        
        Optimizer.zero_grad()
        torch.cuda.empty_cache()
        print("Epoch {}: loss {:.3f}, using time {:.3f}".format(epoch,epoch_loss,epoch_train_time))
        if epoch >= 0:
            if epoch !=0:
                # save(Model,train_ill,test_ill,entid2data,epoch)
                save(Model,train_ill_target,test_ill_target,entid2data_target,epoch)
                
            # test(Model,train_ill,entid2data,TEST_BATCH_SIZE,context="EVAL IN TRAIN SET")
            test(Model, test_ill, entid2data, TEST_BATCH_SIZE, context="EVAL IN TEST SET:")
            test(Model, test_ill_target, entid2data_target, TEST_BATCH_SIZE, context="EVAL IN TARGET TEST SET:")


def test(Model,ent_ill,entid2data,batch_size,context = ""):
    print("-----test start-----")
    start_time = time.time()
    print(context)
    Model.eval()
    with torch.no_grad():
        ents_1 = [e1 for e1,e2 in ent_ill]
        ents_2 = [e2 for e1,e2 in ent_ill]

        emb1 = []
        for i in range(0,len(ents_1),batch_size):
            batch_ents_1 = ents_1[i: i+batch_size]
            batch_emb_1 = entlist2emb(Model,batch_ents_1,entid2data,CUDA_NUM).detach().cpu().tolist()
            emb1.extend(batch_emb_1)
            del batch_emb_1

        emb2 = []
        for i in range(0,len(ents_2),batch_size):
            batch_ents_2 = ents_2[i: i+batch_size]
            batch_emb_2 = entlist2emb(Model,batch_ents_2,entid2data,CUDA_NUM).detach().cpu().tolist()
            emb2.extend(batch_emb_2)
            del batch_emb_2

        print("Cosine similarity of basic bert unit embedding res:")
        res_mat = cos_sim_mat_generate(emb1,emb2,batch_size,cuda_num=CUDA_NUM)
        score,top_index = batch_topk(res_mat,batch_size,topn = TOPK,largest=True,cuda_num=CUDA_NUM)
        hit_res(top_index)
    print("test using time: {:.3f}".format(time.time()-start_time))
    print("--------------------")


def save(Model,train_ill,test_ill,entid2data,epoch_num):
    print("Model {} save in: ".format(epoch_num), MODEL_SAVE_PATH_target + MODEL_SAVE_PREFIX_target + "model_epoch_" + str(epoch_num) + '.p')
    Model.eval()
    torch.save(Model.state_dict(),MODEL_SAVE_PATH_target + MODEL_SAVE_PREFIX_target + "model_epoch_" + str(epoch_num) + '.p')
    other_data = [train_ill,test_ill,entid2data]
    pickle.dump(other_data,open(MODEL_SAVE_PATH_target + MODEL_SAVE_PREFIX_target + 'other_data.pkl',"wb"))
    print("Model {} save end.".format(epoch_num))


#,Train_gene_target,entid2data_target
def ent_align_train(Model,Criterion,Optimizer,Train_gene,entid2data, Train_gene_target,entid2data_target):
    start_time = time.time()
    all_loss = 0
    step = 0
    Model.train()
    print("lllllllll")
    print(len(entid2data))
    print(len(entid2data_target))

    for sour, tar in zip(Train_gene, Train_gene_target):
        
        pe1s,pe2s,ne1s,ne2s  = sour[0], sour[1], sour[2], sour[3]
        pe1s_target,pe2s_target,ne1s_target,ne2s_target = tar[0], tar[1], tar[2], tar[3]
        
        Optimizer.zero_grad()
        
        
        pos_emb1 = entlist2emb(Model,pe1s,entid2data,cuda_num=CUDA_NUM)
        pos_emb2 = entlist2emb(Model,pe2s,entid2data,cuda_num=CUDA_NUM)
       
        batch_length = pos_emb1.shape[0]
        pos_score = F.pairwise_distance(pos_emb1,pos_emb2,p=1,keepdim=True)#L1 distance
        
        sorted_pos_score, indices = torch.sort(pos_score, dim=0)
        # print(indices[:int(len(indices)*0.8)])
        choose_pos = indices[:int(len(indices)*0.5)]
        # # print(pos_emb1[choose_pos])
       

        neg_emb1 = entlist2emb(Model,ne1s,entid2data,cuda_num=CUDA_NUM)
        neg_emb2 = entlist2emb(Model,ne2s,entid2data,cuda_num=CUDA_NUM)
        neg_score = F.pairwise_distance(neg_emb1,neg_emb2,p=1,keepdim=True)
        
        sorted_neg_score, indices_n = torch.sort(neg_score, dim=0, descending=True)
        choose_neg = indices_n[:int(len(indices_n)*0.8)]
        
        

        label_y = -torch.ones(pos_score.shape).cuda(CUDA_NUM) #pos_score < neg_score
        batch_loss = Criterion(pos_score , neg_score , label_y )
        del pos_score
        del neg_score
        del label_y
        
        #target
    
        # print("###############################")
        pos_emb1_target = entlist2emb(Model,pe1s_target,entid2data_target,cuda_num=CUDA_NUM)
        pos_emb2_target = entlist2emb(Model,pe2s_target,entid2data_target,cuda_num=CUDA_NUM)
        batch_length_target = pos_emb1_target.shape[0]
        pos_score_target = F.pairwise_distance(pos_emb1_target,pos_emb2_target,p=1,keepdim=True)#L1 distance
       
        
        

        neg_emb1_target = entlist2emb(Model,ne1s_target,entid2data_target,cuda_num=CUDA_NUM)
        neg_emb2_target = entlist2emb(Model,ne2s_target,entid2data_target,cuda_num=CUDA_NUM)
        neg_score_target = F.pairwise_distance(neg_emb1_target,neg_emb2_target,p=1,keepdim=True)
       

        label_y_target = -torch.ones(pos_score_target.shape).cuda(CUDA_NUM) #pos_score < neg_score
        batch_loss_target = Criterion(pos_score_target , neg_score_target , label_y_target)
        del pos_score_target
        del neg_score_target
        del label_y_target
        
        # print(pos_emb1[choose_pos].shape)
        # print(pos_emb1.shape)
        
        #tranfer loss
        
        transfer_loss1 = MMDLoss()(pos_emb1,pos_emb1_target)
        transfer_loss2 = MMDLoss()(pos_emb2,pos_emb2_target)
        transfer_loss3 = MMDLoss()(neg_emb1,neg_emb1_target)
        transfer_loss4 = MMDLoss()(neg_emb2,neg_emb2_target)
        
        
        # transfer_loss1 = MMDLoss()(pos_emb1[choose_pos].squeeze(1),pos_emb1_target) 
        # transfer_loss2 = MMDLoss()(pos_emb2[choose_pos].squeeze(1),pos_emb2_target)
            
        # transfer_loss3 = MMDLoss()(neg_emb1[choose_neg].squeeze(1),neg_emb1_target)
        # transfer_loss4 = MMDLoss()(neg_emb2[choose_neg].squeeze(1),neg_emb2_target)
       
        
        
        del pos_emb1_target
        del pos_emb2_target
        del pos_emb1
        del pos_emb2
        
        
        del neg_emb1_target
        del neg_emb2_target
        del neg_emb1
        del neg_emb2
#         print(ent1)
        
#         for ent in ent1:
#             print(ent)
#             emb = entlist2emb(Model,ent,entid2data,cuda_num=CUDA_NUM)
#             del emb

            
#         for ent in ent1_target:
#             print(ent)
#             emb = entlist2emb(Model,ent,entid2data_target,cuda_num=CUDA_NUM)
#             del emb
        
        
        
        
        # print(ent1)
        # print(ent1_target)
        #transfer neighbour loss
        
        def get_transloss(ent_s, ent_t):
            trans_loss = 0
            for n_s in ent_s: 
                # print("1")
                for n_t in ent_t: 
                    # print("2")
                    nei_emb_s = entlist2emb(Model,n_s,entid2data,cuda_num=CUDA_NUM)
                    nei_emb_t = entlist2emb(Model,n_t,entid2data_target,cuda_num=CUDA_NUM)
                    t_loss = MMDLoss()(nei_emb_s,nei_emb_t).cpu().item()
                    trans_loss += t_loss
                    
                    del nei_emb_s
                    del nei_emb_t
                    del t_loss
             
            return trans_loss
        # step += 1  
        # print(step)
        # transfer_loss5 = get_transloss(ent1, ent1_target)
        # transfer_loss6 = get_transloss(ent2, ent2_target)   
        
        # transfer_loss7 = get_transloss(nei1_n, nei1_n_target)
        # transfer_loss8 = get_transloss(nei2_n, nei2_n_target)  
        

        
        # print("p2n_nei....")
        # Batch_loss =  batch_loss_target
        Batch_loss =  batch_loss +  batch_loss_target   + transfer_loss3  +  transfer_loss4 
        # + transfer_loss3 +  transfer_loss4 + transfer_loss7 +  transfer_loss8  
        # Batch_loss = batch_loss + batch_loss_target + transfer_loss3 +  transfer_loss4 +  transfer_loss7 +  transfer_loss8 
        
        Batch_loss.backward()
        Optimizer.step()

        all_loss += Batch_loss.item() * batch_length
    
    '''
    
    for pe1s,pe2s,ne1s,ne2s in Train_gene:
        
        Optimizer.zero_grad()
        pos_emb1 = entlist2emb(Model,pe1s,entid2data,cuda_num=CUDA_NUM)
        pos_emb2 = entlist2emb(Model,pe2s,entid2data,cuda_num=CUDA_NUM)
        batch_length = pos_emb1.shape[0]
        pos_score = F.pairwise_distance(pos_emb1,pos_emb2,p=1,keepdim=True)#L1 distance
        del pos_emb1
        del pos_emb2

        neg_emb1 = entlist2emb(Model,ne1s,entid2data,cuda_num=CUDA_NUM)
        neg_emb2 = entlist2emb(Model,ne2s,entid2data,cuda_num=CUDA_NUM)
        neg_score = F.pairwise_distance(neg_emb1,neg_emb2,p=1,keepdim=True)
        del neg_emb1
        del neg_emb2

        label_y = -torch.ones(pos_score.shape).cuda(CUDA_NUM) #pos_score < neg_score
        batch_loss = Criterion( pos_score , neg_score , label_y )
        del pos_score
        del neg_score
        del label_y
        
        batch_loss.backward()
        Optimizer.step()

        all_loss += batch_loss.item() * batch_length
    '''
    
    all_using_time = time.time()-start_time
    return all_loss,all_using_time









