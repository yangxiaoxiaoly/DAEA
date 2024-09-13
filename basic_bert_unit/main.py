import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW
import torch.optim as optim
import random


from Read_data_func import read_data
from Param import *
from Basic_Bert_Unit_model import Basic_Bert_Unit_model
from Batch_TrainData_Generator import Batch_TrainData_Generator
from train_func import train
import numpy as np


def fixed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)



def main():
    #read data
    print("start load data....")
    ent_ill, train_ill, test_ill, \
    index2rel, index2entity, rel2index, entity2index, \
    ent2data, rel_triples_1, rel_triples_2, index2ent1, index2ent2 = read_data(DATA_PATH, DES_DICT_PATH)
    print("---------------------------------------")
    
    
    #get the neighbours
    def get_neigh(ent, triples):
        nei = {}
        for e in ent:
            nei_h = [tri[2] for tri in triples if tri[0] == e]
            nei_t = [tri[0] for tri in triples if tri[2] == e]
            nei_all = set(nei_h + nei_t)
            nei[e] = nei_all
        return nei
        
    random.shuffle(ent_ill)
    train_ill = ent_ill
    # train_ill = random.sample(ent_ill, int(len(ent_ill) * TRAIN_ILL_RATE))
    # test_ill = list(set(ent_ill) - set(train_ill))

    #model
    # Model = Basic_Bert_Unit_model(MODEL_INPUT_DIM,MODEL_OUTPUT_DIM)
    # Model.cuda(CUDA_NUM)

    print("all entity ILLs num:",len(ent_ill))
    print("rel num:",len(index2rel))
    print("ent num:",len(index2entity))
    print("triple1 num:",len(rel_triples_1))
    print("triple2 num:",len(rel_triples_2))

    #get train/test_ill
    if RANDOM_DIVIDE_ILL:
        #get train/test_ILLs by random divide all entity ILLs
        print("Random divide train/test ILLs!")
        random.shuffle(ent_ill)
        train_ill = random.sample(ent_ill, int(len(ent_ill) * TRAIN_ILL_RATE))
        test_ill = list(set(ent_ill) - set(train_ill))
        print("train ILL num: {}, test ILL num: {}".format(len(train_ill), len(test_ill)))
        print("train ILL | test ILL num:", len(set(train_ill) | set(test_ill)))
        print("train ILL & test ILL num:", len(set(train_ill) & set(test_ill)))
    else:
        #get train/test ILLs from file.
        print("get train/test ILLs from file \"sup_pairs\", \"ref_pairs\" !")
        print("train ILL num: {}, test ILL num: {}".format(len(train_ill), len(test_ill)))
        print("train ILL | test ILL:", len(set(train_ill) | set(test_ill)))
        print("train ILL & test ILL:", len(set(train_ill) & set(test_ill)))

    # Criterion = nn.MarginRankingLoss(MARGIN,size_average=True)
    # Optimizer = AdamW(Model.parameters(),lr=LEARNING_RATE)

    ent1 = [e1 for e1,e2 in ent_ill]
    ent2 = [e2 for e1,e2 in ent_ill]
    
    ent1_nei = get_neigh(ent1, rel_triples_1)
    ent2_nei = get_neigh(ent2, rel_triples_2)

    #training data generator(can generate batch-size training data)
    # Train_gene = Batch_TrainData_Generator(train_ill, ent1, ent2,index2entity,batch_size=14,neg_num=NEG_NUM, index2ent1, index2ent2)
    #agrold
    # Train_gene = Batch_TrainData_Generator(train_ill, ent1, ent2,index2entity, index2ent1, index2ent2, ent1_nei, ent2_nei, batch_size=14,neg_num=NEG_NUM)
    #doremus
    Train_gene = Batch_TrainData_Generator(train_ill, ent1, ent2,index2entity, index2ent1, index2ent2, ent1_nei, ent2_nei,ent1_div=2, ent2_div=2, batch_size=TRAIN_BATCH_SIZE,neg_num=NEG_NUM)
    
    


    # train(Model,Criterion,Optimizer,Train_gene,train_ill,test_ill,ent2data)
    
    print("#####################################")
    #load target data................
    #read data
    print("start load data....")
    ent_ill_target, train_ill_target, test_ill_target, \
    index2rel_target, index2entity_target, rel2index_target, entity2index_target, \
    ent2data_target, rel_triples_1_target, rel_triples_2_target, index2ent1_target, index2ent2_target = read_data(DATA_PATH_target, DES_DICT_PATH_target)
    print("---------------------------------------")
    

    #model
    Model = Basic_Bert_Unit_model(MODEL_INPUT_DIM,MODEL_OUTPUT_DIM)
    Model.cuda(CUDA_NUM)

    print("all entity ILLs num:",len(ent_ill_target))
    print("rel num:",len(index2rel_target))
    print("ent num:",len(index2entity_target))
    print("triple1 num:",len(rel_triples_1_target))
    print("triple2 num:",len(rel_triples_2_target))

    #get train/test_ill
    if RANDOM_DIVIDE_ILL:
        #get train/test_ILLs by random divide all entity ILLs
        print("Random divide train/test ILLs!")
        random.shuffle(ent_ill_target)
        train_ill_target = random.sample(ent_ill_target, int(len(ent_ill_target) * TRAIN_ILL_RATE))
        test_ill_target = list(set(ent_ill_target) - set(train_ill_target))
        
        
        # #begin more little train data
        # train_ill_target = train_ill_target[:1875]
        # #end   
        
        
        print("train ILL num: {}, test ILL num: {}".format(len(train_ill_target), len(test_ill_target)))
        print("train ILL | test ILL num:", len(set(train_ill_target) | set(test_ill_target)))
        print("train ILL & test ILL num:", len(set(train_ill_target) & set(test_ill_target)))
    else:
        #get train/test ILLs from file.
        print("get train/test ILLs from file \"sup_pairs\", \"ref_pairs\" !")
        print("train ILL num: {}, test ILL num: {}".format(len(train_ill_target), len(test_ill_target)))
        print("train ILL | test ILL:", len(set(train_ill_target) | set(test_ill_target)))
        print("train ILL & test ILL:", len(set(train_ill_target) & set(test_ill_target)))

    Criterion = nn.MarginRankingLoss(MARGIN,size_average=True)
    Optimizer = AdamW(Model.parameters(),lr=LEARNING_RATE)

    ent1_target = [e1 for e1,e2 in ent_ill_target]
    ent2_target = [e2 for e1,e2 in ent_ill_target]
    
    ent1_nei_target = get_neigh(ent1_target, rel_triples_1_target)
    ent2_nei_target = get_neigh(ent2_target, rel_triples_2_target)
    
    # print('ddd')
    # print(train_ill_target)
    # print('ok')
    # print('ddd')

    #training data generator(can generate batch-size training data)
    #agrold
    # Train_gene_target = Batch_TrainData_Generator(train_ill_target, ent1_target, ent2_target,index2entity_target, index2ent1_target, index2ent2_target, ent1_nei_target, ent2_nei_target, batch_size=11,neg_num=NEG_NUM)
    #doremus
    Train_gene_target = Batch_TrainData_Generator(train_ill_target*6, ent1_target, ent2_target,index2entity_target, index2ent1_target, index2ent2_target, ent1_nei_target, ent2_nei_target, ent1_div=11, ent2_div=6, batch_size=TRAIN_BATCH_SIZE_TARGET,neg_num=NEG_NUM)


    # train(Model,Criterion,Optimizer,Train_gene,train_ill,test_ill,ent2data)  
    train(Model,Criterion,Optimizer,Train_gene,train_ill, test_ill, ent2data,Train_gene_target,train_ill_target,test_ill_target,ent2data_target)
    
    
    
    


if __name__ == '__main__':
    fixed(SEED_NUM)
    main()