import torch
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from model import *
from torch import nn



ps1_index = torch.tensor([source_kg.ent1[e] for e in source_kg.pair1])
ps2_index = torch.tensor([source_kg.ent2[e] for e in source_kg.pair2])

pt1_index = torch.tensor([target_kg.ent1[e] for e in target_kg.pair1])
pt2_index = torch.tensor([target_kg.ent2[e] for e in target_kg.pair2])


s1_w = source_kg.ent1_emb[ps1_index]
s2_w = source_kg.ent2_emb[ps2_index]
t1_w = target_kg.ent1_emb[pt1_index]
t2_w = target_kg.ent2_emb[pt2_index]


s1 = sou1[ps1_index]
s2 = sou2[ps2_index]

t1 = tar1[pt1_index]
t2 = tar2[pt2_index]





def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(p.log(), m, reduction='batchmean') + F.kl_div(q.log(), m, reduction='batchmean'))


# calculate semantics  similarity

prob_a1_w = s1_w.mean(dim=0)
prob_b1_w = t1_w.mean(dim=0)

prob_a2_w = s2_w.mean(dim=0)
prob_b2_w = t2_w.mean(dim=0)

# Normalize the distributions to sum to 1
prob_a1_w /= prob_a1_w.sum()
prob_b1_w /= prob_b1_w.sum()

prob_a2_w /= prob_a2_w.sum()
prob_b2_w /= prob_b2_w.sum()


epsilon = 1e-10
prob_a1_w = torch.clamp(prob_a1_w, min=epsilon)
prob_b1_w = torch.clamp(prob_b1_w, min=epsilon)
prob_a2_w = torch.clamp(prob_a2_w, min=epsilon)
prob_b2_w = torch.clamp(prob_b2_w, min=epsilon)

jsd1_w = js_divergence(prob_a1_w, prob_b1_w)
js_distance1_w = torch.sqrt(jsd1_w)

jsd2_w = js_divergence(prob_a2_w, prob_b2_w)
js_distance2_w = torch.sqrt(jsd2_w)

# print("Semantic Jensen-Shannon Divergence:", jsd1_w.item()*100, jsd2_w.item()*100)
print("Semantic Jensen-Shannon Distance:", js_distance1_w.item()*100, js_distance2_w.item()*100)

# calculate structure similarity

prob_a1 = s1.mean(dim=0)
prob_b1 = t1.mean(dim=0)

prob_a2 = s2.mean(dim=0)
prob_b2 = t2.mean(dim=0)

# Normalize the distributions to sum to 1
prob_a1 /= prob_a1.sum()
prob_b1 /= prob_b1.sum()

prob_a2 /= prob_a2.sum()
prob_b2 /= prob_b2.sum()


epsilon = 1e-10
prob_a1 = torch.clamp(prob_a1, min=epsilon)
prob_b1 = torch.clamp(prob_b1, min=epsilon)
prob_a2 = torch.clamp(prob_a2, min=epsilon)
prob_b2 = torch.clamp(prob_b2, min=epsilon)

jsd1 = js_divergence(prob_a1, prob_b1)
js_distance1 = torch.sqrt(jsd1)

jsd2 = js_divergence(prob_a2, prob_b2)
js_distance2 = torch.sqrt(jsd2)

# print("Structure Jensen-Shannon Divergence:", jsd1.item()*100, jsd2.item()*100)
print("Structure Jensen-Shannon Distance:", js_distance1.item()*100, js_distance2.item()*100)
