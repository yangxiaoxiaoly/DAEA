## DAEA: Enhancing Entity Alignment in Real-World Knowledge Graphs Through Multi-Source Domain Adaptation
Through Multi-Source Domain Adaptation

This is code and datasets for DAEA

### Dependencies

- Python 3 (tested on 3.9.0)
- Pytorch (tested on 2.2.1)
- [transformers](https://github.com/huggingface/transformers) (tested on 2.1.1)
- torch_geometric (tested on 2.5.3)

### Dataset

**DBP15K** and **Real-World Data**

All the data can be downloaded from https://drive.google.com/file/d/1OXzwPve7PT1jRHfdZajNro2MYM4I7Lc0/view?usp=drive_link.

Initial DBP15K datasets are from **JAPE**(<https://github.com/nju-websoft/JAPE>). 

**Description data**

- `data/dbp15k/2016-10-des_dict`: A dictionary storing entity descriptions, which can be loaded by pickle.load(). The description of the entity is extracted from **DBpedia**(<https://wiki.dbpedia.org/downloads-2016-10>)
- `data/agrold_en/2024-06-03-agrold`: A dictionary storing entity descriptions, which can be loaded by pickle.load(). The description of the entity is extracted from https://github.com/EnsiyehRaoufi/Create_Input_Data_to_EA_Models.
- `data/doremus_en/2024-02-12-doremus`: A dictionary storing entity descriptions, which can be loaded by pickle.load(). The description of the entity is extracted from https://github.com/EnsiyehRaoufi/Create_Input_Data_to_EA_Models.



### How to Run

The model runs in three steps: 

#### 1. Multi-Source KGs selection

The source and target datasets path can be changed in multisource/param.py.

``` shell
cd multisource/
python main.py
```

#### 2. Fine-tune Basic BERT Unit followed  https://github.com/kosugi11037/bert-int

To fine-tune the Basic BERT Unit, use: 

```shell
cd basic_bert_unit/
python main.py
```

Note that `basic_bert_unit/Param.py` is the config file.

The obtained Basic BERT Unit and some other data will be stored in:  `../Save_model`

#### 3. Run BERT-based Interaction Model followed https://github.com/kosugi11037/bert-int

(Note that when running the BERT-based Interaction model, the parameters of the Basic BERT Unit model will **be fixed**.)

To extract the similarity features and run the BERT-base Interaction Model, use:

```shell
cd ../interaction_model/
python clean_attribute_data.py
python get_entity_embedding.py
python get_attributeValue_embedding.py
python get_neighView_and_desView_interaction_feature.py
python get_attributeView_interaction_feature.py
python interaction_model.py
```

Or directly use:

```shell
cd ../interaction_model/
bash run.sh
```

Note that `interaction_model/Param.py` is the config file.

