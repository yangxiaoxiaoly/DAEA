import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from load import *
from param import *

source_fold = source_fold
source_lang = source_lang

target_fold = target_fold
target_lang = target_lang

# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, heads=1):
#         super(GAT, self).__init__()
#         self.conv1 = GATConv(in_channels, out_channels, heads=heads)
    
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         return F.elu(x)
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.gat2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return x

    
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=10):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, edge_index):
        # Assume first half of 'edge_index' is positive samples
        anchor, negative = embeddings[edge_index[0]], embeddings[edge_index[1]]
        # negative = embeddings[torch.randint(0, embeddings.size(0), edge_index[0].size())]

        # positive_distance = F.pairwise_distance(anchor, positive)
        negative_distance = F.pairwise_distance(anchor, negative)

        losses = F.relu(- negative_distance + self.margin)
        return losses.mean()

model = GAT(in_channels=300, out_channels=300) 


loss_func = ContrastiveLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train(data):
    model.train()
    optimizer.zero_grad()
    embeddings = model(data)
    loss = loss_func(embeddings, data.edge_index)
    loss.backward()
    optimizer.step()
    return loss


source_kg = Getdata(source_fold, source_lang)
target_kg = Getdata(target_fold, target_lang)



source_kg1 = Data(x=source_kg.ent1_embran, edge_index=source_kg.edge1.t().contiguous())
source_kg2 = Data(x=source_kg.ent2_embran, edge_index=source_kg.edge2.t().contiguous())

target_kg1 = Data(x=target_kg.ent1_embran, edge_index=target_kg.edge1.t().contiguous())
target_kg2 = Data(x=target_kg.ent2_embran, edge_index=target_kg.edge2.t().contiguous())


# graph = Data(x=x, edge_index=edge_index)

for epoch in range(200):
    loss = train(source_kg1)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

for epoch in range(200):
    loss = train(source_kg2)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

for epoch in range(200):
    loss = train(target_kg1)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')
for epoch in range(200):
    loss = train(target_kg2)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    sou1 = model(source_kg1)
    sou2 = model(source_kg2)
    tar1 = model(target_kg1)
    tar2 = model(target_kg2)

    

    

    
    
