import torch
import json
from tqdm import tqdm

class Getdata:
    def __init__(self, fold, lang):
        self.fold = fold
        self.lang = lang
        self.ent1 = self.load_entid(f"{fold}{lang}ent_ids_1")
        self.ent2 = self.load_entid(f"{fold}{lang}ent_ids_2")
        self.triples1 = self.load_triples(f"{fold}{lang}triples_1")
        self.triples2 = self.load_triples(f"{fold}{lang}triples_2")
        self.embed = self.load_json(f"{fold}{lang}{lang.split('_')[0]}_vectorList.json")
        self.ent1_emb = self.embed[torch.tensor(list(self.ent1.keys()))]
        self.ent2_emb = self.embed[torch.tensor(list(self.ent2.keys()))]
        
        self.ent1_embran = torch.randn(len(self.ent1), 300)
        self.ent2_embran = torch.randn(len(self.ent2), 300)
        self.edge1 = self.get_edge(self.ent1, self.triples1)
        self.edge2 = self.get_edge(self.ent2, self.triples2)
        self.pairs = self.load_pairs(f"{fold}{lang}sup_pairs")
        
        self.pair1 = [p[0] for p in self.pairs]
        self.pair2 = [p[1] for p in self.pairs]
        
        

    def load_entid(self, path):
        ent = {}
        with open(path, 'r', encoding="utf-8") as f:
            for c, line in enumerate(f.readlines()):
                t = tuple(line.strip().split())
                ent[int(t[0])] = c
        return ent

    def load_json(self, path):
        with open(path, mode='r', encoding='utf-8') as f:
            embedding_list = json.load(f)
        input_embeddings = torch.tensor(embedding_list, dtype=torch.float32)
        return input_embeddings

    def load_triples(self, path):
        tr = []
        with open(path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                t = tuple(line.strip().split())
                if len(t) != 3:
                    print('===')
                tr.append((int(t[0]), int(t[1]), int(t[2])))
        return tr
    def load_pairs(self, path):
        pairs = []
        with open(path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                t = tuple(line.strip().split())               
                pairs.append((int(t[0]), int(t[1])))
        return pairs

    
    def get_edge(self, ent, triples):
        edge = {}
        for tri in triples:
            if (ent[tri[0]], ent[tri[2]]) not in edge:
                edge[(ent[tri[0]], ent[tri[2]])] = 1
            else:
                edge[(ent[tri[0]], ent[tri[2]])] += 1
        return torch.tensor(list(edge.keys()),dtype=torch.long)
    

    

        

        


