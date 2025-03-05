import torch
from torch import nn
import math
from .utils import inverse_sigmoid

class QueryInteractionBase(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out):
        super().__init__()
        self._build_layers(dim_in, hidden_dim, dim_out)
        self._reset_parameters()

    def _build_layers(self, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()


def pos2posemb(pos, num_pos_feats=64, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    posemb = pos[..., None] / dim_t
    posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
    return posemb


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt


class QueryInteractionModule(QueryInteractionBase):
    def __init__(self,dim_in, hidden_dim, dim_out):
        super().__init__(dim_in, hidden_dim, dim_out)

    def _build_layers(self,dim_in, hidden_dim, dim_out):
        dropout = 0.1

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)
        """
        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)
        """
        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        """
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)
        """
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        """
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)
        """
        self.activation = nn.ReLU(True)

    def _update_track_embedding(self, track_queries):
        """
        if track_queries['track_ref_points_b'] is None:
           print('track_ref_points_b is none')
        query_pos = pos2posemb(track_queries['track_ref_points_b'])
        #print('query_pos:',query_pos.shape)
        query_feat = track_queries['embeds_b']
        out_embed = track_queries['embeds']
        """
        query_pos = pos2posemb(inverse_sigmoid(track_queries['pred_boxes_2']))
        query_feat = track_queries['embeds_2_b']
        out_embed = track_queries['embeds_2']

        q = k = query_pos + out_embed
        #print('q:',q.shape)

        tgt = out_embed
        #print('tgt shape:',tgt.shape)
        #print('q shape:',q.shape)
        #print('k shape:',k.shape)
        tgt2 = self.self_attn(q, k, value=tgt)[0][:, 0]
        #print('pasa tgt2')
        #print('tgt2 shape:',tgt2.shape)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        """
        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_queries['position'] = query_pos
        """

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        #track_queries['embeds'] = query_feat
        track_queries['embeds_2'] = query_feat

        return track_queries

    def forward(self, track_queries):
        track_queries = self._update_track_embedding(track_queries)
        return track_queries
