import dgl
import torch
from dgl.nn.pytorch import HeteroGraphConv, GATConv
from han.my_han import HAN

g = dgl.heterograph({
    ('A', 'AB', 'B'): ([0, 1, 2], [1, 2, 3]),
    ('B', 'BA', 'A'): ([1, 2, 3], [0, 1, 2])})
new_g = dgl.metapath_reachable_graph(g, ['AB', 'BA'])
new_g.edges(order='eid')

# 假设你有两种模态的数据分别为 data1 和 data2
# 假设 data1 的长度为 250，data2 的长度为 50
# 假设每个数据都是一个长为 192 的嵌入向量
num_nodes_data1 = 250
num_nodes_data2 = 50
embedding_dim = 192

data1 = torch.randn(num_nodes_data1, embedding_dim)
data2 = torch.randn(num_nodes_data2, embedding_dim)

# 假设你有一个共同的边索引 edge_index
# edge_index 是一个包含两行的张量，每一列表示一条边的两个节点的索引
i2a_edge_index = torch.tensor([
    [0, 0, 1, 1, 2, 249],  # 假设这里包含了所有的边的起始节点索引
    [5, 12, 7, 2, 9, 49],  # 假设这里包含了所有的边的目标节点索引
])
a2i_edge_index = torch.tensor([
    [0, 12, 7, 9, 49],  # 假设这里包含了所有的边的起始节点索引
    [0, 0, 1, 2, 249],  # 假设这里包含了所有的边的目标节点索引
])

# 注意：边索引应该包含两行，其中第一行是源节点的索引，第二行是目标节点的索引
# 假设 edge_index 表示 data1 中的节点和 data2 中的节点之间的连接关系

# 创建一个空的异构图 H
H = dgl.heterograph({
    ('image', 'i2a', 'audio'): (i2a_edge_index[0], i2a_edge_index[1]),
    ('audio', 'a2i', 'image'): (a2i_edge_index[0], a2i_edge_index[1])
})

# 添加节点特征
H.nodes['image'].data['feature'] = data1
H.nodes['audio'].data['feature'] = data2

# mg = dgl.metapath_reachable_graph(H, ['i2a', 'a2i'])
# print(mg.srcdata['feature'].shape)
# print(mg.dstdata['feature'].shape)
# print(mg.number_of_nodes())
# gat = GATConv(192, 192, 2)
# mg = dgl.add_self_loop(mg)
# output = gat(mg, (mg.srcdata['feature'], mg.dstdata['feature']))

batch_H = dgl.batch([H, H])

i_han = HAN([['i2a', 'a2i'], ['i2a', 'a2i', 'i2a', 'a2i']], 192, 192, 192, [4, 4, 4], 0.0)
i_han_output = i_han(batch_H, batch_H.ndata['feature']['image'])
a_han = HAN([['a2i', 'i2a'], ['a2i', 'i2a', 'a2i', 'i2a']], 192, 192, 192, [4, 4, 4], 0.0)
a_han_output = a_han(batch_H, batch_H.ndata['feature']['audio'])

# unbatch
i_han_output = dgl.unbatch(i_han_output)
pass
