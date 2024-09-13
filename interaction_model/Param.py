"""
hyper-parameters:
"""
CUDA_NUM = 0 #GPU num
LANG = 'fr' #language 'zh'/'ja'/'fr'


target = "fr"


ENTITY_NEIGH_MAX_NUM = 50 # max sampling neighbor num of entity
ENTITY_ATTVALUE_MAX_NUM = 50 #max sampling attributeValue num of entity
KERNEL_NUM = 21
SEED_NUM = 11037
CANDIDATE_NUM = 50 # candidate number

BATCH_SIZE = 5 # train batch size
NEG_NUM = 5 # negative sampling num
LEARNING_RATE = 5e-4 # learning rate
MARGIN = 1 # margin
EPOCH_NUM = 200 # train epoch num

INTERACTION_MODEL_SAVE_PATH = "../Save_model/interaction_model_{}en.bin".format(LANG) #interaction model save path.

#load model(base_bert_unit_model) path
BASIC_BERT_UNIT_MODEL_SAVE_PATH = "../Save_model_trans_des_ori_p_100/"
BASIC_BERT_UNIT_MODEL_SAVE_PREFIX = "realworld_{}en".format(LANG)
LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM = 4
BASIC_BERT_UNIT_MODEL_OUTPUT_DIM = 300


BASIC_BERT_UNIT_MODEL_SAVE_PREFIX_target = "realworld_{}en".format(target)

#load data path
DATA_PATH = r"../data/{}_en/".format(LANG)
DATA_PATH_target = r"../data/{}_en/".format(target)


#load source
#candidata_save_path
TRAIN_CANDIDATES_PATH = DATA_PATH + 'train_candidates.pkl'
TEST_CANDIDATES_PATH = DATA_PATH + 'test_candidates.pkl'

#entity embedding and attributeValue embedding save path.
ENT_EMB_PATH = DATA_PATH + '{}_emb_{}.pkl'.format(BASIC_BERT_UNIT_MODEL_SAVE_PREFIX,LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM)
ATTRIBUTEVALUE_EMB_PATH = DATA_PATH + 'attributeValue_embedding.pkl'
ATTRIBUTEVALUE_LIST_PATH = DATA_PATH + 'attributeValue_list.pkl' #1-1 match to attributeValue embedding.

#(candidate) entity_pairs save path.
ENT_PAIRS_PATH = DATA_PATH + 'ent_pairs.pkl' #[(e1,ea),(e1,eb)...]

#interaction feature save filepath name
NEIGHBORVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + 'neighbor_view_similarity_feature.pkl' #1-1 match to entity_pairs
ATTRIBUTEVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + 'attribute_similarity_feature.pkl' #1-1 match to entity_pairs
DESVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + 'des_view_similarity_feature.pkl' #1-1 match to entity_pairs


#load target
########################################

#candidata_save_path
TRAIN_CANDIDATES_PATH_target = DATA_PATH_target + 'train_candidates.pkl'
TEST_CANDIDATES_PATH_target = DATA_PATH_target + 'test_candidates.pkl'

#entity embedding and attributeValue embedding save path.
ENT_EMB_PATH_target = DATA_PATH_target + '{}_emb_{}.pkl'.format(BASIC_BERT_UNIT_MODEL_SAVE_PREFIX_target,LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM)
ATTRIBUTEVALUE_EMB_PATH_target = DATA_PATH_target + 'attributeValue_embedding.pkl'
ATTRIBUTEVALUE_LIST_PATH_target = DATA_PATH_target + 'attributeValue_list.pkl' #1-1 match to attributeValue embedding.

#(candidate) entity_pairs save path.
ENT_PAIRS_PATH_target = DATA_PATH_target + 'ent_pairs.pkl' #[(e1,ea),(e1,eb)...]

#interaction feature save filepath name
NEIGHBORVIEW_SIMILARITY_FEATURE_PATH_target = DATA_PATH_target + 'neighbor_view_similarity_feature.pkl' #1-1 match to entity_pairs
ATTRIBUTEVIEW_SIMILARITY_FEATURE_PATH_target = DATA_PATH_target + 'attribute_similarity_feature.pkl' #1-1 match to entity_pairs
DESVIEW_SIMILARITY_FEATURE_PATH_target = DATA_PATH_target + 'des_view_similarity_feature.pkl' #1-1 match to 