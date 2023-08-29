import dgl
import torch
from dgl.nn.pytorch import HeteroGraphConv, GATConv
from torch import nn

from models.mmh_vig.han.my_han import HAN


class AV_HAN(torch.nn.Module):
    def __init__(self, input_size=192, hidden_size=192, output_size=192, num_layers=1, dropout=0.0):
        super(AV_HAN, self).__init__()
        self.i_han = HAN(
            [
                ['i2a', 'a2i'],
                # ['i2a', 'a2i', 'i2a', 'a2i']
            ], input_size, hidden_size, output_size, [1]*num_layers, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.a_han = HAN(
            [
                ['a2i', 'i2a'],
                # ['a2i', 'i2a', 'a2i', 'i2a']
            ], input_size, hidden_size, output_size, [1]*num_layers, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, batch_features, edge_indexes):
        total_len = batch_features.shape[1]
        audio_len = 48
        image_len = total_len - audio_len
        batch_image_features = batch_features[:, :image_len]
        batch_audio_features = batch_features[:, image_len:]
        edge_indexes = edge_indexes.permute(1, 2, 3, 0)
        edge_indexes = edge_indexes.reshape(edge_indexes.shape[0], edge_indexes.shape[1] * edge_indexes.shape[2], edge_indexes.shape[3])
        graphs = []
        batch_size = batch_image_features.shape[0]
        for i in range(batch_size):
            image_features = batch_image_features[i]
            audio_features = batch_audio_features[i]
            edge_index = edge_indexes[i]
            edge_index_x = edge_index[:, 0].unsqueeze(1)
            edge_index_y = edge_index[:, 1].unsqueeze(1)
            edge_index = torch.cat([edge_index_y, edge_index_x], dim=1)
            i2a_mask = (edge_index[:, 0] < image_len) & (edge_index[:, 1] >= image_len)
            a2i_mask = (edge_index[:, 0] >= image_len) & (edge_index[:, 1] < image_len)
            i2a_edge_index = edge_index[i2a_mask]
            a2i_edge_index = edge_index[a2i_mask]

            # Convert the results to torch tensors
            i2a_edge_index = i2a_edge_index - torch.tensor([0, image_len], device=i2a_edge_index.device)
            a2i_edge_index = a2i_edge_index - torch.tensor([image_len, 0], device=a2i_edge_index.device)
            i2a_edge_index = torch.cat((i2a_edge_index, torch.tensor([[len(image_features) - 1, len(audio_features) - 1]], device=i2a_edge_index.device)))
            # a2i_edge_index = torch.cat((a2i_edge_index, torch.tensor([[len(audio_features) - 1, len(image_features) - 1]], device=a2i_edge_index.device)))

            g = dgl.heterograph({
                ('image', 'i2a', 'audio'): (i2a_edge_index[:, 0], i2a_edge_index[:, 1]),
                ('audio', 'a2i', 'image'): (a2i_edge_index[:, 0], a2i_edge_index[:, 1])
            }, device=image_features.device)
            g.nodes['image'].data['h'] = image_features
            g.nodes['audio'].data['h'] = audio_features
            graphs.append(g)

        batch_graph = dgl.batch(graphs)
        i_h = self.i_han(batch_graph, batch_graph.ndata['h']['image'])
        i_h = self.norm1(i_h)
        i_h = i_h.reshape(batch_size, -1, i_h.shape[-1])
        a_h = self.a_han(batch_graph, batch_graph.ndata['h']['audio'])
        a_h = a_h.reshape(batch_size, -1, a_h.shape[-1])
        a_h = self.norm2(a_h)
        output = torch.cat((i_h, a_h), dim=1)
        return output
