import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.sampling import RandomWalkNeighborSampler

from han.model_hetero import HANLayer, HAN
from dgl.nn.pytorch import HeteroGraphConv

# 创建异构图1
g1 = dgl.heterograph({
    ('image', 'i2a', 'audio'): (torch.tensor([0]), torch.tensor([1])),
    ('audio', 'a2i', 'image'): (torch.tensor([1]), torch.tensor([2]))
})
g1.nodes['image'].data['h'] = torch.tensor([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]])
g1.nodes['audio'].data['h'] = torch.tensor([[0.3, 0.4], [0.3, 0.5]])

# 创建异构图2
g2 = dgl.heterograph({
    ('image', 'i2a', 'audio'): (torch.tensor([0]), torch.tensor([0])),
    ('audio', 'a2i', 'image'): (torch.tensor([0]), torch.tensor([1]))
})
g2.nodes['image'].data['h'] = torch.tensor([[1.1, 1.2], [1.0, 1.2]])
g2.nodes['audio'].data['h'] = torch.tensor([[1.3, 1.4]])


# 将两个异构图组合成一个批次
batched_graph = dgl.batch([g1, g2])

sampler = RandomWalkNeighborSampler(
                    G=batched_graph,
                    num_traversals=1,
                    termination_prob=0,
                    num_random_walks=3,
                    num_neighbors=3,
                    metapath=['i2a', 'a2i'],
                )
# frontier = sampler(1)
# block = dgl.to_block(frontier, 1)
# print(block)

# new_g = dgl.metapath_reachable_graph(batched_graph, ['i2a', 'a2i'])
# print(new_g.edges(order='eid'))

model = HAN(
    meta_paths=[['i2a', 'a2i']],
    in_size=2,
    hidden_size=2,
    out_size=2,
    num_heads=[2],
    dropout=0.0
)

batched_graph = dgl.to_block(batched_graph)
output = model(batched_graph, batched_graph.ndata['h'])
# output = model(batched_graph, (batched_graph.nodes['image'].data['h'], batched_graph.nodes['audio'].data['h']))
# output = model(batched_graph, (batched_graph.nodes['image'].data['h'], batched_graph.nodes['audio'].data['h']))
