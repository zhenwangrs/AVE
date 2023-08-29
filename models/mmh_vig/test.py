import dgl
import torch
from torch import nn
from models.mmh_vig import mm_vig
import sys
from transformers import RobertaModel, RobertaTokenizer

from collections import defaultdict


# def find_meta_paths(edge_index, node_types):
#     graph = defaultdict(list)
#     for u, v in edge_index:
#         graph[u].append(v)
#         graph[v].append(u)
#
#     meta_paths = []
#     for u in graph:
#         for v in graph[u]:
#             for w in graph[v]:
#                 if w != u:
#                     path = [u, v, w]
#                     types = [node_types[x] for x in path]
#                     if types == ['image', 'audio', 'image'] or types == ['audio', 'image', 'audio']:
#                         meta_paths.append(path)
#
#     return meta_paths
#
#
# edge_index = [(1, 2), (2, 3), (3, 4), (4, 1)]
# node_types = {1: 'image', 2: 'audio', 3: 'image', 4: 'audio'}
#
# print(find_meta_paths(edge_index, node_types))

if __name__ == '__main__':
    # data_dict = {
    #        ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
    #        ('user', 'follows', 'topic'): (torch.tensor([1, 1]), torch.tensor([1, 2])),
    #        ('user', 'plays', 'game'): (torch.tensor([0, 3]), torch.tensor([3, 4]))
    #     }
    # g = dgl.heterograph(data_dict)
    # print(g)

    vig = mm_vig.vig_ti_224_gelu()
    img = torch.randn(4, 3, 224, 224)
    audio = torch.randn(4, 3, 96, 128)
    out = vig(img, audio)
    pass
