from model_train_test_func import *
from Param import *

def main():
    print("----------------interaction model--------------------")
    cuda_num = CUDA_NUM
    print("GPU num {}".format(cuda_num))
    #print("ko~ko~da~yo~")

    # read other data from bert unit model(train ill/test ill/eid2data)
    # (These files were saved during the training of basic bert unit)
    bert_model_other_data_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + 'other_data.pkl'
    train_ill, test_ill, eid2data = pickle.load(open(bert_model_other_data_path, "rb"))
    print("train_ill num: {} /test_ill num:{} / train_ill & test_ill num: {}".format(len(train_ill),len(test_ill), len(set(train_ill) & set(test_ill) )))


    #(candidate) entity pairs
    entity_pairs = pickle.load(open(ENT_PAIRS_PATH, "rb"))

    #interaction features
    nei_features = pickle.load(open(NEIGHBORVIEW_SIMILARITY_FEATURE_PATH, "rb")) #neighbor-view interaction similarity feature
    att_features = pickle.load(open(ATTRIBUTEVIEW_SIMILARITY_FEATURE_PATH,'rb')) #attribute-view interaction similarity feature
    des_features = pickle.load(open(DESVIEW_SIMILARITY_FEATURE_PATH, "rb")) #description/name-view interaction similarity feature
    train_candidate = pickle.load(open(TRAIN_CANDIDATES_PATH, "rb"))
    test_candidate = pickle.load(open(TEST_CANDIDATES_PATH, "rb"))
    all_features = [] #[nei-view cat att-view cat des/name-view]
    for i in range(len(entity_pairs)):
        all_features.append(nei_features[i]+ att_features[i]+ des_features[i])# 42 concat 42 concat 1.
        # all_features.append(att_features[i]+ des_features[i])
    print("All features embedding shape: ", np.array(all_features).shape)
    
    entpair2f_idx = {entpair: feature_idx for feature_idx, entpair in enumerate(entity_pairs)}
    Train_gene = Train_index_generator(train_ill, train_candidate, entpair2f_idx,neg_num=NEG_NUM, batch_size=BATCH_SIZE)

    
    '''
    #load target
    ################
    # read other data from bert unit model(train ill/test ill/eid2data)
    # (These files were saved during the training of basic bert unit)
    bert_model_other_data_path_target = BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX_target + 'other_data.pkl'
    train_ill_target, test_ill_target, eid2data_target = pickle.load(open(bert_model_other_data_path_target, "rb"))
    print("train_ill num: {} /test_ill num:{} / train_ill & test_ill num: {}".format(len(train_ill_target),len(test_ill_target), len(set(train_ill_target) & set(test_ill_target) )))


    #(candidate) entity pairs
    entity_pairs_target = pickle.load(open(ENT_PAIRS_PATH_target, "rb"))

    #interaction features
    nei_features_target = pickle.load(open(NEIGHBORVIEW_SIMILARITY_FEATURE_PATH_target, "rb")) #neighbor-view interaction similarity feature
    att_features_target = pickle.load(open(ATTRIBUTEVIEW_SIMILARITY_FEATURE_PATH_target,'rb')) #attribute-view interaction similarity feature
    des_features_target = pickle.load(open(DESVIEW_SIMILARITY_FEATURE_PATH_target, "rb")) #description/name-view interaction similarity feature
    train_candidate_target = pickle.load(open(TRAIN_CANDIDATES_PATH_target, "rb"))
    test_candidate_target = pickle.load(open(TEST_CANDIDATES_PATH_target, "rb"))
    all_features_target = [] #[nei-view cat att-view cat des/name-view]
    for i in range(len(entity_pairs_target)):
        all_features_target.append(nei_features_target[i]+ att_features_target[i]+ des_features_target[i])
        # all_features_target.append(att_features_target[i]+ des_features_target[i])# 42 concat 42 concat 1.
    print("All features embedding shape: ", np.array(all_features_target).shape)
    
    entpair2f_idx_target = {entpair: feature_idx for feature_idx, entpair in enumerate(entity_pairs_target)}
    Train_gene_target = Train_index_generator(train_ill_target, train_candidate_target, entpair2f_idx_target,neg_num=NEG_NUM, batch_size=5)
    
   
    ########################
    '''
   
    

    
    Model = MlP(42 * 2 + 1,11).cuda(cuda_num)
    
    Optimizer = optim.Adam(Model.parameters(), lr=LEARNING_RATE)
    Criterion = nn.MarginRankingLoss(margin=MARGIN, size_average=True)

    
    
    
    
    
    
    #train
    train(Model, Optimizer, Criterion, Train_gene, all_features, test_candidate, test_ill,
          entpair2f_idx, epoch_num=EPOCH_NUM, eval_num=10, cuda_num=cuda_num, test_topk=50)
    
    # train(Model, Optimizer, Criterion, Train_gene, all_features, test_candidate, test_ill,
    #       entpair2f_idx, Train_gene_target, all_features_target, test_candidate_target, test_ill_target,
    #       entpair2f_idx_target, epoch_num=EPOCH_NUM, eval_num=10, cuda_num=cuda_num, test_topk=50)

    #save
    torch.save(Model, open(INTERACTION_MODEL_SAVE_PATH, "wb"))


if __name__ == '__main__':
    fixed(SEED_NUM)
    main()