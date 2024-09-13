import numpy as np
import torch
import torch.nn as nn
from Param import *
import time


class Batch_TrainData_Generator(object):
    def __init__(self,train_ill,ent_ids1,ent_ids2,index2entity, index2ent1, index2ent2,ent1_nei, ent2_nei,ent1_div, ent2_div, batch_size,neg_num):
        self.ent_ill = train_ill
        self.ent_ids1 = ent_ids1
        self.ent_ids2 = ent_ids2
        self.batch_size = batch_size
        self.neg_num = neg_num
        
        # self.ent_size = ent_size
        
        self.iter_count = 0
        self.index2entity = index2entity
        
        self.ent1_nei = ent1_nei
        self.ent2_nei = ent2_nei
        
        self.ent1 = list(index2ent1.keys())
        self.ent2 = list(index2ent2.keys())
        # self.all_ent = list(index2entity.keys())
        
#         ent1_all = []
#         l1 = len(self.ent1)
#         l1_en =  l1//ent1_div
#         for i in range(ent1_div):
#             ent1_all.append(self.ent1[i * l1//ent1_div : (i+1) * l1//ent1_div])
            
#         ent2_all = []
#         l2 = len(self.ent2)
#         l2_en =  l2//ent2_div
#         for i in range(ent2_div):
#             ent2_all.append(self.ent1[i * l2//ent2_div : (i+1) * l2//ent2_div])
        
#         self.ent1_all = ent1_all
#         self.ent2_all = ent2_all
        
        print("In Batch_TrainData_Generator, train ill num: {}".format(len(self.ent_ill)))
        print("In Batch_TrainData_Generator, ent_ids1 num: {}".format(len(self.ent_ids1)))
        print("In Batch_TrainData_Generator, ent_ids2 num: {}".format(len(self.ent_ids2)))
        # print("In Batch_TrainData_Generator, keys of index2entity num: {}".format(len(self.index2entity)))
        
        




    def train_index_gene(self,candidate_dict):
        """
        generate training data (entity_index).
        """
        train_index = [] #training data
        candid_num = 999999
        for ent in candidate_dict:
            candid_num = min(candid_num,len(candidate_dict[ent]))
            candidate_dict[ent] = np.array(candidate_dict[ent])
        for pe1,pe2 in self.ent_ill:
            for _ in range(self.neg_num):
                if np.random.rand() <= 0.5:
                    #e1
                    ne1 = candidate_dict[pe2][np.random.randint(candid_num)]
                    ne2 = pe2
                else:
                    ne1 = pe1
                    ne2 = candidate_dict[pe1][np.random.randint(candid_num)]
                #same check
                if pe1!=ne1 or pe2!=ne2:
                    train_index.append([pe1,pe2,ne1,ne2])
        np.random.shuffle(train_index)
        self.train_index = train_index
        self.batch_num = int( np.ceil( len(self.train_index) * 1.0 / self.batch_size ) )
        
        print("dddddddddllllllllll")
        print("train data length:")
        print(len(train_index))
        



    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_count < self.batch_num:
            batch_index = self.iter_count
            self.iter_count += 1

            batch_data = self.train_index[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]

            pe1s = [pe1 for pe1,pe2,ne1,ne2 in batch_data]
            pe2s = [pe2 for pe1,pe2,ne1,ne2 in batch_data]
            ne1s = [ne1 for pe1,pe2,ne1,ne2 in batch_data]
            ne2s = [ne2 for pe1,pe2,ne1,ne2 in batch_data]
            
            def get_nei(ent, neigh):
                nei_list = []
                for e in ent:
                    nei_e = []
                    if len(neigh[e]) > 20:
                        nei_e = list(neigh[e])[:20]
                    else:
                        nei_e = list(neigh[e])       
                    nei_list.append(nei_e)
                return nei_list
            
#             nei1 = get_nei(pe1s,self.ent1_nei)
#             nei2 = get_nei(pe2s,self.ent2_nei)
            
#             nei1_n = get_nei(ne1s,self.ent1_nei)
#             nei2_n = get_nei(ne2s,self.ent2_nei)
            
            # ent1 = self.ent1[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]
            # ent2 = self.ent2[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]
            
            ent1 = []
            ent2 = []

#             for item in self.ent1_all:
#                 # ent1.append(item[batch_index * 15 : (batch_index + 1) * 15])
#                 ent1.append(item[batch_index * self.batch_size : (batch_index + 1) * self.batch_size])
            
#             for item in self.ent2_all:
#                 # ent2.append(item[batch_index * 15 : (batch_index + 1) * 15])
#                 ent2.append(item[batch_index * self.batch_size : (batch_index + 1) * self.batch_size])
            
            # return pe1s,pe2s,ne1s,ne2s, ent1, ent2
            return pe1s,pe2s,ne1s,ne2s

        else:
            del self.train_index
            self.iter_count = 0
            raise StopIteration()
