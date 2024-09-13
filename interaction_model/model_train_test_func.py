import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from utils import *
import copy
from torch.nn import init
from mmd import *
from coral import *
from adv import *

class MlP(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(MlP, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim, True)
        self.dense2 = nn.Linear(hidden_dim, 1, True)
        init.xavier_normal_(self.dense1.weight)
        init.xavier_normal_(self.dense2.weight)
    def forward(self,features):
        x = self.dense1(features)#[B,h]
        x = F.relu(x)
        x = self.dense2(x)#[B,1]
        x = F.tanh(x)
        x = torch.squeeze(x,1)#[B]
        return x

class Train_index_generator(object):
    def __init__(self, train_ill, train_candidate, entpair2f_idx, neg_num, batch_size):
        self.train_ill = train_ill
        self.train_candidate = copy.deepcopy(train_candidate)
        self.entpair2f_idx = entpair2f_idx
        self.iter_count = 0
        self.batch_size = batch_size
        self.neg_num = neg_num
        print("In Train_batch_index_generator, train_ILL num : {}".format(len(self.train_ill)))
        print("In Train_batch_index_generator, Batch size: {}".format(self.batch_size))
        print("In Train_batch_index_generator, Negative sampling num: {}".format(self.neg_num))
        for e in self.train_candidate.keys():
            self.train_candidate[e] = np.array(self.train_candidate[e])
        self.train_pair_indexs, self.batch_num = self.train_pair_index_gene()

    def train_pair_index_gene(self):
        """
        generate training data (entity_index).
        """
        train_pair_indexs = []
        for pe1, pe2 in self.train_ill:          
            
            neg_indexs = np.random.randint(len(self.train_candidate[pe1]),size=self.neg_num)
            ne2_list = self.train_candidate[pe1][neg_indexs].tolist()
            new_ne2_list = ne2_list.copy()
            for ne2 in ne2_list:
                if ne2 == pe2:
                    new_ne2_list.remove(ne2)
                    ne2 = new_ne2_list[0]                      
                ne1 = pe1
                train_pair_indexs.append((pe1, pe2, ne1, ne2))
                #(pe1,pe2) is aligned entity pair, (ne1,ne2) is negative sample
        np.random.shuffle(train_pair_indexs)
        np.random.shuffle(train_pair_indexs)
        np.random.shuffle(train_pair_indexs)
        
        # print("all data:")
        # print("len(train_pair_indexs):",len(train_pair_indexs))
        
        batch_num = int(np.ceil(len(train_pair_indexs) * 1.0 / self.batch_size))
        # batch_num = 71
        # self.iter_count = 0
        # print("batchnum", batch_num)
        return train_pair_indexs, batch_num

    def __iter__(self):
        return self

    def __next__(self):
        
        if self.iter_count < self.batch_num:
            batch_index = self.iter_count
            # print("print(self.iter_count): ",batch_index)
            
            batch_ids = self.train_pair_indexs[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
            pos_pairs = [(pe1, pe2) for pe1, pe2, ne1, ne2 in batch_ids]
            neg_pairs = [(ne1, ne2) for pe1, pe2, ne1, ne2 in batch_ids]

            pos_f_ids = [self.entpair2f_idx[pair_id] for pair_id in pos_pairs]
            neg_f_ids = [self.entpair2f_idx[pair_id] for pair_id in neg_pairs]
            self.iter_count += 1
            return pos_f_ids, neg_f_ids
        else:
            # print("else:",self.iter_count)
            self.iter_count = 0
            self.train_pair_indexs, self.batch_num = self.train_pair_index_gene()
            # print("fffffff")
            # return None
            raise StopIteration

def one_step_train(Model, Optimizer, Criterion, Train_gene, f_emb, cuda_num):
    epoch_loss = 0
    for pos_f_ids, neg_f_ids in Train_gene:
        Optimizer.zero_grad()
        pos_feature = f_emb[torch.LongTensor(pos_f_ids)].cuda(cuda_num)
        neg_feature = f_emb[torch.LongTensor(neg_f_ids)].cuda(cuda_num)
        p_score = Model(pos_feature)
        n_score = Model(neg_feature)
        p_score = p_score.unsqueeze(-1)#[B,1]
        n_score = n_score.unsqueeze(-1)#[B,1]
        batch_size = p_score.shape[0]
        label_y = torch.ones(p_score.shape).cuda(cuda_num) #if y == 1 mean: p_score should ranked higher.
        batch_loss = Criterion(p_score, n_score, label_y)  #p_score > n_score
        epoch_loss += batch_loss.item() * batch_size
        batch_loss.backward()
        Optimizer.step()
    return epoch_loss


def test(Model, test_candidate, test_ill, entpair2f_idx, f_emb, batch_size, cuda_num, test_topk):
    test_ill_set = set(test_ill)
    test_pairs = []#all candidate entity pairs of Test set.
    for e1 in [a for a, b in test_ill]:
        for e2 in test_candidate[e1]:
            test_pairs.append((e1, e2))
    isin_test_ill_set_num = sum([pair in test_ill_set for pair in test_pairs])
    print("all test entity pair num {}/ max align entity pair num: {}".format(len(test_pairs), isin_test_ill_set_num))
    scores = []
    for start_pos in range(0, len(test_pairs), batch_size):
        batch_pair_ids = test_pairs[start_pos:start_pos + batch_size]
        batch_f_ids = [entpair2f_idx[pair_idx] for pair_idx in batch_pair_ids]
        batch_features = f_emb[torch.LongTensor(batch_f_ids)].cuda(cuda_num)  # [B,f]
        batch_scores = Model(batch_features)
        batch_scores = batch_scores.detach().cpu().tolist()
        scores.extend(batch_scores)
    assert len(test_pairs) == len(scores)
    # eval
    e1_to_e2andscores = dict()
    for i in range(len(test_pairs)):
        e1, e2 = test_pairs[i]
        score = scores[i]
        if (e1, e2) in test_ill_set:
            label = 1
        else:
            label = 0
        if e1 not in e1_to_e2andscores:
            e1_to_e2andscores[e1] = []
        e1_to_e2andscores[e1].append((e2, score, label))

    all_test_num = len(e1_to_e2andscores.keys()) # test set size.
    result_labels = []
    for e, value_list in e1_to_e2andscores.items():
        v_list = value_list
        v_list.sort(key=lambda x: x[1], reverse=True)
        label_list = [label for e2, score, label in v_list]
        label_list = label_list[:test_topk]
        result_labels.append(label_list)
    result_labels = np.array(result_labels)
    result_labels = result_labels.sum(axis=0).tolist()
    topk_list = []
    for i in range(test_topk):
        nums = sum(result_labels[:i + 1])
        topk_list.append(round(nums / all_test_num, 5))
    print("hit @ 1: {:.5f}    hit @10 : {:.5f}    ".format(topk_list[1 - 1], topk_list[10 - 1]), end="")
    if test_topk >= 25:
        print("hit @ 25: {:.5f}    ".format(topk_list[25 - 1]), end="")
    if test_topk >= 50:
        print("hit @ 50: {:.5f}    ".format(topk_list[50 - 1]), end="")
    print("")
    # MRR
    MRR = 0
    for i in range(len(result_labels)):
        MRR += (1 / (i + 1)) * result_labels[i]
    MRR /= all_test_num
    print("MRR:", MRR)


def train(Model, Optimizer, Criterion, Train_gene, f_emb_list, test_candidate, test_ill,
          entpair2f_idx, epoch_num, eval_num, cuda_num, test_topk):
    feature_emb = torch.FloatTensor(f_emb_list)
    print("start training interaction model!")
    for epoch in range(epoch_num):
        start_time = time.time()
        epoch_loss = one_step_train(Model, Optimizer, Criterion, Train_gene, feature_emb, cuda_num)
        print("Epoch {} loss {:.4f} using time {:.3f}".format(epoch, epoch_loss, time.time() - start_time))
        if (epoch + 1) % eval_num == 0 and epoch != 0 :
            start_time = time.time()
            test(Model, test_candidate, test_ill, entpair2f_idx, feature_emb, 2048, cuda_num, test_topk)
            print("test using time {:.3f}".format(time.time() - start_time))            
            
            
'''
#transfer in step2
def one_step_train(Model, Optimizer, Criterion, Train_gene, f_emb, Train_gene_target, f_emb_target, cuda_num):
    epoch_loss = 0
    # epoch_loss_target = 0
    # epoch_loss_trans = 0
    step = 0
    alpha = 0.1
    
    pos_f_ids =[i for i, j in Train_gene]
    
    neg_f_ids =[j for i, j in Train_gene]
    
    pos_f_ids_target =[i for i, j in Train_gene_target]
    neg_f_ids_target =[j for i, j in Train_gene_target]
    while step < 71: # 
        # print("step:",step)
        Optimizer.zero_grad()
        # pos_f_ids, neg_f_ids = Train_gene.__next__()
        # print(len(pos_f_ids))
        # pos_f_ids_target, neg_f_ids_target = Train_gene_target.__next__()
        # Train_gene.__next__()
        # Train_gene_target.__next__()
        
        pos_feature = f_emb[torch.LongTensor(pos_f_ids[step])].cuda(cuda_num)
        neg_feature = f_emb[torch.LongTensor(neg_f_ids[step])].cuda(cuda_num)
        p_score = Model(pos_feature)
        n_score = Model(neg_feature)
        p_score = p_score.unsqueeze(-1)#[B,1]
        n_score = n_score.unsqueeze(-1)#[B,1]
        
        
        pos_feature_target = f_emb_target[torch.LongTensor(pos_f_ids_target[step])].cuda(cuda_num)
        neg_feature_target = f_emb_target[torch.LongTensor(neg_f_ids_target[step])].cuda(cuda_num)
        p_score_target = Model(pos_feature_target)
        n_score_target = Model(neg_feature_target)
        p_score_target = p_score_target.unsqueeze(-1)#[B,1]
        n_score_target = n_score_target.unsqueeze(-1)#[B,1]

        batch_size = p_score.shape[0]
        
        label_y = torch.ones(p_score.shape).cuda(cuda_num) #if y == 1 mean: p_score should ranked higher.
        batch_loss = Criterion(p_score, n_score, label_y)  #p_score > n_score
        
        batch_size_target = p_score_target.shape[0]
        
        label_y_target = torch.ones(p_score_target.shape).cuda(cuda_num) #if y == 1 mean: p_score should ranked higher.
        batch_loss_target = Criterion(p_score_target, n_score_target, label_y_target)  #p_score > n_score
        
        
        
        
        transfer_loss = AdversarialLoss()(torch.cat((pos_feature, neg_feature), dim=0), torch.cat((pos_feature_target, neg_feature_target), dim=0))
        
        # print("loss:")
        
        # print(type(transfer_loss))
        # print(batch_loss)
        # Batch_loss = batch_loss + batch_loss_target + alpha * transfer_loss
        Batch_loss = 0*batch_loss + 1*batch_loss_target + 0*transfer_loss
        
        Batch_loss.backward()
        Optimizer.step()
       
        step = step + 1
        epoch_loss += Batch_loss
        
#     for pos_f_ids, neg_f_ids in Train_gene:
#         Optimizer.zero_grad()
#         pos_feature = f_emb[torch.LongTensor(pos_f_ids)].cuda(cuda_num)
#         neg_feature = f_emb[torch.LongTensor(neg_f_ids)].cuda(cuda_num)
#         p_score = Model(pos_feature)
#         n_score = Model(neg_feature)
#         p_score = p_score.unsqueeze(-1)#[B,1]
#         n_score = n_score.unsqueeze(-1)#[B,1]
        
#         batch_size = p_score.shape[0]
        
#         label_y = torch.ones(p_score.shape).cuda(cuda_num) #if y == 1 mean: p_score should ranked higher.
#         batch_loss = Criterion(p_score, n_score, label_y)  #p_score > n_score
#         epoch_loss += batch_loss.item() * batch_size

#         batch_loss.backward()
#         Optimizer.step()
        
#     for pos_f_ids, neg_f_ids in Train_gene_target:
#         Optimizer.zero_grad()
#         pos_feature = f_emb_target[torch.LongTensor(pos_f_ids)].cuda(cuda_num)
#         neg_feature = f_emb_target[torch.LongTensor(neg_f_ids)].cuda(cuda_num)
#         p_score = Model(pos_feature)
#         n_score = Model(neg_feature)
#         p_score = p_score.unsqueeze(-1)#[B,1]
#         n_score = n_score.unsqueeze(-1)#[B,1]
#         batch_size = p_score.shape[0]
#         label_y = torch.ones(p_score.shape).cuda(cuda_num) #if y == 1 mean: p_score should ranked higher.
#         batch_loss = Criterion(p_score, n_score, label_y)  #p_score > n_score
#         epoch_loss_target += batch_loss.item() * batch_size
        
#         batch_loss.backward()
#         Optimizer.step()  
#     for pos_f_ids, neg_f_ids in Train_gene: 
#         for pos_f_ids_target, neg_f_ids_target in Train_gene_target:
#             Optimizer.zero_grad()
#             pos_feature = f_emb[torch.LongTensor(pos_f_ids)].cuda(cuda_num)
#             neg_feature = f_emb[torch.LongTensor(neg_f_ids)].cuda(cuda_num)
#             p_score = Model(pos_feature)
#             n_score = Model(neg_feature)
#             p_score = p_score.unsqueeze(-1)#[B,1]
#             n_score = n_score.unsqueeze(-1)#[B,1]
            
#             pos_feature_target = f_emb_target[torch.LongTensor(pos_f_ids_target)].cuda(cuda_num)
#             neg_feature_target = f_emb_target[torch.LongTensor(neg_f_ids_target)].cuda(cuda_num)
#             p_score_target = Model(pos_feature_target)
#             n_score_target = Model(neg_feature_target)
#             p_score_target = p_score_target.unsqueeze(-1)#[B,1]
#             n_score_target = n_score_target.unsqueeze(-1)#[B,1]
            
#             p_dis = torch.cdist(p_score, p_score_target, p=1)
#             n_dis = torch.cdist(n_score, n_score_target, p=1)
            
#             p_dis = p_dis.unsqueeze(-1)#[B,1]
#             n_dis = n_dis.unsqueeze(-1)#[B,1]
            
    
#             batch_size = p_dis.shape[0]
#             label_y = torch.ones(p_dis.shape).cuda(cuda_num) #if y == 1 mean: p_score should ranked higher.
#             batch_loss = Criterion(p_dis , n_dis  , label_y)  #p_score > n_score
#             epoch_loss_trans += batch_loss.item() * batch_size
        
#             batch_loss.backward()
#             Optimizer.step() 
    
        
    return epoch_loss 


def test(Model, test_candidate, test_ill, entpair2f_idx, f_emb, test_candidate_target, test_ill_target, entpair2f_idx_target, f_emb_target, batch_size, cuda_num, test_topk):
    
    # Model = torch.load('../Save_model/interaction_model_agrolden.bin')
    # Model.eval() 
    test_ill_set = set(test_ill)
    test_pairs = []#all candidate entity pairs of Test set.
    for e1 in [a for a, b in test_ill]:
        for e2 in test_candidate[e1]:
            test_pairs.append((e1, e2))
    isin_test_ill_set_num = sum([pair in test_ill_set for pair in test_pairs])
    print("all test entity pair num {}/ max align entity pair num: {}".format(len(test_pairs), isin_test_ill_set_num))
    scores = []
    for start_pos in range(0, len(test_pairs), batch_size):
        batch_pair_ids = test_pairs[start_pos:start_pos + batch_size]
        batch_f_ids = [entpair2f_idx[pair_idx] for pair_idx in batch_pair_ids]
        batch_features = f_emb[torch.LongTensor(batch_f_ids)].cuda(cuda_num)  # [B,f]
        batch_scores = Model(batch_features)
        batch_scores = batch_scores.detach().cpu().tolist()
        scores.extend(batch_scores)
    assert len(test_pairs) == len(scores)
    # eval
    e1_to_e2andscores = dict()
    for i in range(len(test_pairs)):
        e1, e2 = test_pairs[i]
        score = scores[i]
        if (e1, e2) in test_ill_set:
            label = 1
        else:
            label = 0
        if e1 not in e1_to_e2andscores:
            e1_to_e2andscores[e1] = []
        e1_to_e2andscores[e1].append((e2, score, label))

    all_test_num = len(e1_to_e2andscores.keys()) # test set size.
    result_labels = []
    for e, value_list in e1_to_e2andscores.items():
        v_list = value_list
        v_list.sort(key=lambda x: x[1], reverse=True)
        label_list = [label for e2, score, label in v_list]
        label_list = label_list[:test_topk]
        result_labels.append(label_list)
    result_labels = np.array(result_labels)
    result_labels = result_labels.sum(axis=0).tolist()
    topk_list = []
    for i in range(test_topk):
        nums = sum(result_labels[:i + 1])
        topk_list.append(round(nums / all_test_num, 5))
    print("hit @ 1: {:.5f}    hit @10 : {:.5f}    ".format(topk_list[1 - 1], topk_list[10 - 1]), end="")
    if test_topk >= 25:
        print("hit @ 25: {:.5f}    ".format(topk_list[25 - 1]), end="")
    if test_topk >= 50:
        print("hit @ 50: {:.5f}    ".format(topk_list[50 - 1]), end="")
    print("")
    # MRR
    MRR = 0
    for i in range(len(result_labels)):
        MRR += (1 / (i + 1)) * result_labels[i]
    MRR /= all_test_num
    print("MRR:", MRR)
    
    #target.......
    ##################
    test_ill_set_target = set(test_ill_target)
    test_pairs_target = []#all candidate entity pairs of Test set.
    for e1 in [a for a, b in test_ill_target]:
        for e2 in test_candidate_target[e1]:
            test_pairs_target.append((e1, e2))
    isin_test_ill_set_num_target = sum([pair in test_ill_set_target for pair in test_pairs_target])
    print("all test entity pair num {}/ max align entity pair num: {}".format(len(test_pairs_target), isin_test_ill_set_num_target))
    scores_target = []
    for start_pos in range(0, len(test_pairs_target), batch_size):
        batch_pair_ids = test_pairs_target[start_pos:start_pos + batch_size]
        batch_f_ids = [entpair2f_idx_target[pair_idx] for pair_idx in batch_pair_ids]
        batch_features = f_emb_target[torch.LongTensor(batch_f_ids)].cuda(cuda_num)  # [B,f]
        batch_scores = Model(batch_features)
        batch_scores = batch_scores.detach().cpu().tolist()
        scores_target.extend(batch_scores)
    assert len(test_pairs_target) == len(scores_target)
    # eval
    e1_to_e2andscores_target = dict()
    for i in range(len(test_pairs_target)):
        e1, e2 = test_pairs_target[i]
        score = scores_target[i]
        if (e1, e2) in test_ill_set_target:
            label = 1
        else:
            label = 0
        if e1 not in e1_to_e2andscores_target:
            e1_to_e2andscores_target[e1] = []
        e1_to_e2andscores_target[e1].append((e2, score, label))

    all_test_num_target = len(e1_to_e2andscores_target.keys()) # test set size.
    result_labels_target = []
    for e, value_list in e1_to_e2andscores_target.items():
        v_list = value_list
        v_list.sort(key=lambda x: x[1], reverse=True)
        label_list = [label for e2, score, label in v_list]
        label_list = label_list[:test_topk]
        result_labels_target.append(label_list)
    result_labels_target = np.array(result_labels_target)
    result_labels_target = result_labels_target.sum(axis=0).tolist()
    topk_list_target = []
    for i in range(test_topk):
        nums = sum(result_labels_target[:i + 1])
        topk_list_target.append(round(nums / all_test_num_target, 5))
    print("hit @ 1: {:.5f}    hit @10 : {:.5f}    ".format(topk_list_target[1 - 1], topk_list_target[10 - 1]), end="")
    if test_topk >= 25:
        print("hit @ 25: {:.5f}    ".format(topk_list_target[25 - 1]), end="")
    if test_topk >= 50:
        print("hit @ 50: {:.5f}    ".format(topk_list_target[50 - 1]), end="")
    print("")
    # MRR
    MRR_target = 0
    for i in range(len(result_labels_target)):
        MRR_target += (1 / (i + 1)) * result_labels_target[i]
    MRR_target /= all_test_num_target
    print("MRR_target:", MRR_target)


def train(Model, Optimizer, Criterion, Train_gene, f_emb_list, test_candidate, test_ill,
          entpair2f_idx, Train_gene_target, f_emb_list_target, test_candidate_target, test_ill_target,
          entpair2f_idx_target, epoch_num, eval_num, cuda_num, test_topk):
    
    
    feature_emb = torch.FloatTensor(f_emb_list)
    feature_emb_target = torch.FloatTensor(f_emb_list_target)
    print("start training interaction model!")
    for epoch in range(epoch_num):
        start_time = time.time()
        epoch_loss = one_step_train(Model, Optimizer, Criterion, Train_gene, feature_emb, Train_gene_target, feature_emb_target, cuda_num)
        print("Epoch {} loss {:.4f} using time {:.3f}".format(epoch, epoch_loss, time.time() - start_time))
        
        if (epoch + 1) % eval_num == 0 and epoch != 0 :
            start_time = time.time()
            test(Model, test_candidate, test_ill, entpair2f_idx, feature_emb, test_candidate_target, test_ill_target, entpair2f_idx_target, feature_emb_target, 2048, cuda_num, test_topk)
            print("test using time {:.3f}".format(time.time() - start_time))
'''