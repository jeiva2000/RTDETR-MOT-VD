"""by lyuwenyu
"""

import math 
import copy 
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 

from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob

from .rtdetr_decoder import TransformerDecoderLayer, TransformerDecoder, RTDETRTransformer

from src.core import register

from .qim import QueryInteractionModule

from .transformers import MultiheadAttention as MultiheadAttention_aux

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

import numpy as np
import time
import cv2


#torch.set_printoptions(threshold=10_000)

__all__ = ['RTDETRTransformerMOT_v2']


class segmentation_head(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(segmentation_head,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=4,padding=1)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=4,padding=1)     
        self.conv3=nn.Conv2d(out_channels,out_channels,kernel_size=16,padding=1)    
        self.layers = nn.ModuleList( nn.Conv2d(in_channels,out_channels) if i == 0 else nn.Conv2d(out_channels*(i),out_channels*i+1) for i in range(4) ) 
    
    def forward(self,x):
        for layer in layers:
            x = layer(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4,):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2,)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()


    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            #value_mask = value_mask.astype(value.dtype).unsqueeze(-1) #quitar si la modificacino de abajo no funciona
            value *= value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(
                bs, Len_q, 1, self.num_levels, 1, 2
            ) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))


        output, sampling_values = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        return output, sampling_values

class TransformerDecoderLayerMOT_v2(TransformerDecoderLayer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 deformable=True):
        super().__init__(d_model,n_head,dim_feedforward,dropout,
                         activation,n_levels,n_points)
        self.deformable = deformable
        # cross attention
        if deformable:
           self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        else:
           #self.cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
           self.cross_attn = MultiHeadAttentionBlock(d_model, n_head, dropout=dropout)
           #self.cross_attn = MultiheadAttention_aux(d_model, n_head, dropout=dropout, batch_first=True)

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                return_attn=False):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        # if attn_mask is not None:
        #     attn_mask = torch.where(
        #         attn_mask.to(torch.bool),
        #         torch.zeros_like(attn_mask),
        #         torch.full_like(attn_mask, float('-inf'), dtype=tgt.dtype))

        tgt2, self_attn_weights = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        if self.deformable:
           tgt2, sampling_values = self.cross_attn(\
                   self.with_pos_embed(tgt, query_pos_embed), 
                   reference_points, 
                   memory, 
                   memory_spatial_shapes, 
                   memory_mask)
        else:
           """
           tgt2, attn_weights = self.cross_attn(\
                   self.with_pos_embed(tgt, query_pos_embed),
                   memory,
                   value=memory,
                   attn_mask=memory_mask)
           """
           tgt2, attn_weights = self.cross_attn(\
                   self.with_pos_embed(tgt, query_pos_embed),
                   memory,
                   memory,
                   memory_mask)
            
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        if return_attn:
           return tgt, attn_weights, self_attn_weights
        else:
            if self.deformable:
                return tgt, sampling_values
            else:
                return tgt


class TransformerDecoderMOT_v2(TransformerDecoder):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__(hidden_dim, decoder_layer, num_layers, eval_idx)
    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                num_denoising,
                attn_mask=None,
                memory_mask=None,
                track_queries=None,
                mode=0):

        output = tgt

        dec_out_bboxes = []
        dec_out_logits = []
        sampling_values_list = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)
            output, sampling_values = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed) #add memory_mask
            sampling_values_list.append(sampling_values)
            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), output, sampling_values_list[-1]

class TransformerDecoderMOT_track(TransformerDecoder):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__(hidden_dim, decoder_layer, num_layers, eval_idx)
        self.self_attn = MultiHeadAttentionBlock(hidden_dim,8,0.2)
    def forward(self,
                tgt,
                query_pos_embed,
                ref_points,
                memory,
                mask,
                bbox_head,
                score_head):
        output = tgt
        #print('mask shape:',mask.shape)
        #print('tgt shape:',tgt.shape)
        #print('memory shape:',memory.shape)
        #aux_tgt = torch.cat((tgt,memory),1)
        #_, attention_w = self.self_attn(aux_tgt,aux_tgt,aux_tgt,None)
        #attention_w = attention_w.mean(1)
        #print('attention_w shape:',attention_w.shape)
        #print('attention_w:',attention_w)
        boxes = []
        scores = []
        for i, layer in enumerate(self.layers):
            output, attn_weights, self_attn_weights = layer(output, None, memory,None, None, memory_mask=mask,query_pos_embed=query_pos_embed,return_attn=True)
            #print('output:',output)
            ref_points = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points))
            scores.append(score_head[i](output))
            boxes.append(ref_points)
        return output, boxes[-1], scores[-1], attn_weights, self_attn_weights


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        print('query shape:',query.shape)
        print('key shape:',key.shape)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        #print('scores shape:',attention_scores.shape)
        #if mask is not None:
        #    attention_scores.masked_fill_(mask == 0, -1e9)
        #attention_scores = attention_scores * mask
        attention_scores_logits = attention_scores.clone().mean(1)
        #attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        #attention_scores = torch.sigmoid(attention_scores) #probar
        #print('scores soft:',attention_scores)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores_logits

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x), self.attention_scores

@register
class RTDETRTransformerMOT_v2(RTDETRTransformer):
    __share__ = ['num_classes']
    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True):

        super().__init__(num_classes,hidden_dim,num_queries,position_embed_type,
                         feat_channels,feat_strides,num_levels,num_decoder_points,
                         nhead,num_decoder_layers,dim_feedforward,dropout,activation,
                         num_denoising,label_noise_ratio,box_noise_scale,learnt_init_query,
                         eval_spatial_size,eval_idx,eps,aux_loss)

        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        self.decoder = TransformerDecoderMOT_v2(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        #deformable track
        decoder_layer_track = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)

        #interact queries
        track_decoder_layer_h = TransformerDecoderLayerMOT_v2(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points,deformable=False)
        self.decoder_track = TransformerDecoderMOT_track(hidden_dim, track_decoder_layer_h, num_decoder_layers)
        #self.track_decoder_layer_h = TransformerDecoderLayerMOT_v2(3072, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points,deformable=False)

        #self.proj_mask = MLP(hidden_dim, 2 * hidden_dim, num_queries, num_layers=6)
        self.proj_mask_track = nn.Linear(hidden_dim, hidden_dim, bias=False) # Wq #MLP(hidden_dim, 2 * hidden_dim, 1, num_layers=6)
        self.proj_mask_det =  nn.Linear(hidden_dim, hidden_dim, bias=False) # Wk #MLP(hidden_dim, 2 * hidden_dim, 1, num_layers=6)

        self.proj_mask_query =  nn.Linear(hidden_dim, hidden_dim, bias=False) # Wk #MLP(hidden_dim, 2 * hidden_dim, 1, num_layers=6)

        #self.proj_embeds = MLP(3072+hidden_dim, 2 * hidden_dim, hidden_dim, num_layers=6)
        
        self.proj_embeds = MLP(3072, 2 * hidden_dim, hidden_dim, num_layers=6)

        self.dec_score_head_track = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head_track = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_decoder_layers)
        ])
        bias = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.dec_score_head_track, self.dec_bbox_head_track):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)

    def forward(self, feats, targets=None, track_queries=None, ind_track=True, mode='', limit=2, update_track=False): #limit=10 or 30 -> cuidado con el limite
        # input projection and embedding
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising, 
                    label_noise_ratio=self.label_noise_ratio, 
                    box_noise_scale=self.box_noise_scale, )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits, output, sampling_values_det = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.num_denoising,
            attn_mask=attn_mask,
            #track_queries=track_queries,
            mode=0)

        track_queries['hs'] = output
        sampling_values_det = torch.stack(sampling_values_det)
        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            _ , track_queries['hs'] = torch.split(track_queries['hs'], dn_meta['dn_num_split'], dim=1)
            dn_out_sampling, sampling_values_det = torch.split(sampling_values_det, dn_meta['dn_num_split'], dim=3)
        
        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        
        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        th = 0.5

        with torch.no_grad():
             if self.training:
                ious = generalized_box_iou(box_cxcywh_to_xyxy(targets[0]['boxes']),box_cxcywh_to_xyxy(out_bboxes[-1].squeeze(0)))
                if ious.shape[1] > 0:
                    values, indices = torch.max(ious,dim=1)
                    indices = indices[values > th]
                    mask = torch.zeros(out_logits[-1].shape[1])
                    mask[torch.unique(indices)] = 1
                    mask = mask > 0
                    mask = mask.unsqueeze(0)
                    track_queries['mask_pred'] = mask.clone()
                else:
                    mask = torch.tensor([])
                    track_queries['mask_pred'] = torch.tensor([])
             else:
                scores = F.sigmoid(out_logits[-1])
                mask = torch.max(scores,2)[0] > th
                track_queries['mask_pred'] = mask.clone()

        """
        #generate attention matrix
        print('memory shape:',memory.shape)
        print('spatial shapes:',spatial_shapes)
        print('level start index:',level_start_index)
        aux_memory = memory[:,level_start_index[-1]:memory.shape[1],:]
        print('aux_memory shape:',aux_memory.shape)
        attention_scores = (track_queries['hs'][mask] @ aux_memory.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        #soft_attn = attention_scores.softmax(dim=-1).squeeze(0)
        soft_attn = torch.sigmoid(attention_scores).squeeze(0)
        soft_attn = (soft_attn > 0.8).float()
        aux_image = torch.zeros((640,640,1))
        print('soft_attn shape:',soft_attn.shape)
        for j,soft in enumerate(soft_attn):
            soft = soft.reshape(int(math.sqrt(soft.shape[0])),int(math.sqrt(soft.shape[0])))
            soft = soft.unsqueeze(2)
            print('soft_attn shape:',soft.shape)
            soft = cv2.resize(soft.cpu().detach().numpy()*255,(640,640))
            soft = torch.tensor(soft).unsqueeze(2)
            print('soft shape:',soft.shape)
            aux_image+=soft
        cv2.imwrite('prueba_atencion.jpg',aux_image.cpu().detach().numpy())
        #aux_idxs = torch.nonzero(attention_scores > 0.5)
        #print('aux_idxs:',aux_idxs)
        #print('attn scores shape:',attention_scores.shape)
        """

        if 'embeds_2' in track_queries and track_queries['embeds_2'] is not None and ind_track:
            out_bboxes_2, out_logits_2, output_2, sampling_values_track = self.decoder(
            track_queries['embeds_2'],#prueba
            inverse_sigmoid(track_queries['pred_boxes_2']),
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            0,
            attn_mask=None,
            mode=1)
            sampling_values_track = torch.stack(sampling_values_track)
            track_queries['embeds_2'] = output_2
            track_queries['pred_boxes_2'] = out_bboxes_2[-1]
            track_queries['pred_logits_2'] = out_logits_2[-1]
        if mask.sum() > 0 and ind_track:
            th_assoc = 0.5
            if 'embeds_2' not in track_queries:
                track_queries['embeds_2'] = track_queries['hs'][mask].unsqueeze(0)
                track_queries['pred_boxes_2'] = out_bboxes[-1][mask].unsqueeze(0)
                track_queries['pred_logits_2'] = out_logits[-1][mask].unsqueeze(0)
                if "id_max" not in track_queries:
                   track_queries['ids'] = torch.arange(0,track_queries['embeds_2'].shape[1]).to(track_queries['embeds_2'].device)
                track_queries['id_max'] = torch.max(track_queries["ids"])
                track_queries['delete'] = torch.zeros(track_queries['embeds_2'].shape[1]).to(track_queries['embeds_2'].device)
                track_queries['re_boxes'] = None
                track_queries['re_logits'] = None
            else:
                track_queries['pred_logits_2_b'] = track_queries['pred_logits_2'].clone()
                track_queries['pred_boxes_2_b'] = track_queries['pred_boxes_2'].clone()
                #print('mask shape:',mask.shape)
                #proj_track = self.proj_embeds(torch.cat((track_queries['embeds_2'].squeeze(0),torch.flatten(sampling_values_track.permute(3,0,1,2,4),start_dim=1)),dim=1))
                #proj_det = self.proj_embeds(torch.cat((track_queries['hs'][mask.clone()],torch.flatten(sampling_values_det[:,:,:,mask.clone().squeeze(0),:].permute(3,0,1,2,4),start_dim=1)),dim=1))
                #proj_det = proj_det.unsqueeze(0)
                #proj_track = proj_track.unsqueeze(0)
                #print('proj_det shape:',proj_det.shape)
                #print('proj_track shape:',proj_track.shape)
                #_, boxes, scores, last_attn_weights = self.decoder_track(proj_track,self.query_pos_head(track_queries['pred_boxes_2']),track_queries['pred_boxes_2'],proj_det,None,self.dec_bbox_head_track,self.dec_score_head_track)
                orig_ids = torch.nonzero(mask.clone())[:,1].to('cuda')
                #print('ids originales:',orig_ids)
                #det_samplings = torch.flatten(sampling_values_det[:,:,:,mask.clone().squeeze(0),:].permute(3,0,1,2,4),start_dim=1)
                #det_samplings = self.proj_embeds(det_samplings).unsqueeze(0)
                #proj_det = self.proj_mask_det(track_queries['hs'][mask.clone()])
                #proj_track = self.proj_mask_track(track_queries['embeds_2']).squeeze(0)
                #proj_det = torch.cat((proj_det,proj_track),dim=0)
                #dec_mask = proj_track@proj_det.t()/math.sqrt(256)
                #dec_mask = torch.where(dec_mask > 0, dec_mask, 0.)
                #track_queries['mask'] = dec_mask
                #print('dec mask shape:',dec_mask.shape)
                #print('track embeds shape:',track_queries['embeds_2'].shape)
                #print('det embeds shape:',track_queries['hs'][mask].shape)
                #print('det_samplings shape:',det_samplings.shape)
                #_, boxes, scores, last_attn_weights, self_attn = self.decoder_track(track_queries['embeds_2'],self.query_pos_head(track_queries['pred_boxes_2']),track_queries['pred_boxes_2'],det_samplings,dec_mask,self.dec_bbox_head_track,self.dec_score_head_track)
                #aux_det = torch.cat((track_queries['hs'][mask.clone()].unsqueeze(0),track_queries['embeds_2']),dim=1)
                _, boxes, scores, last_attn_weights, self_attn = self.decoder_track(track_queries['embeds_2'],self.query_pos_head(track_queries['pred_boxes_2']),track_queries['pred_boxes_2'],track_queries['hs'][mask.clone()].unsqueeze(0),None,self.dec_bbox_head_track,self.dec_score_head_track)
                #print('self_attn:',self_attn)
                track_queries['mask'] = last_attn_weights
                #print('last attn weights:',last_attn_weights.requires_grad)
                track_queries['re_boxes'] = boxes
                track_queries['re_logits'] = scores

                ious = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['pred_boxes_2'].squeeze(0)),box_cxcywh_to_xyxy(out_bboxes[-1][mask]))
                aux_ious = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['pred_boxes_2'].squeeze(0)),box_cxcywh_to_xyxy(out_bboxes[-1].squeeze(0)))
                if self.training:
                    sim_idxs = torch.nonzero(ious > th_assoc)[:,1]
                    #aux_ids = torch.nonzero(torch.sigmoid(track_queries['mask']).squeeze(0) > th_assoc)
                else:
                    #print('mask:',torch.sigmoid(track_queries['mask']))
                    #print('mask:',torch.sigmoid(track_queries['mask']).shape)
                    #print('shape mask:',torch.sigmoid(track_queries['mask']).squeeze(0).shape)
                    #print('track querie mask shape:',track_queries['mask'].shape)
                    #print('aux_ious:',aux_ious)
                    #print('track querie mask:',torch.sigmoid(track_queries['mask']))
                    print('ious:',ious)
                    print('embeds shape:',track_queries['embeds_2'].shape)
                    print('dets shape:',track_queries['hs'][mask.clone()].shape)
                    print('track_mask:',track_queries['mask'].shape)
                    #print('track querie mask:',track_queries['mask'][:track_queries['embeds_2'].shape[1],:track_queries['hs'][mask.clone()].shape[0]])
                    #print('attn_weights shape:',last_attn_weights.shape)
                    aux_ids = torch.nonzero(track_queries['mask'].squeeze(0) > th_assoc)
                    #aux_ids = torch.nonzero(track_queries['mask'][:track_queries['embeds_2'].shape[1],:track_queries['hs'][mask.clone()].shape[0]] > th_assoc)                    
                    print('aux_ids:',aux_ids)
                    print('orig_ids:',orig_ids)
                    #aux_ids = torch.nonzero(track_queries['mask'].squeeze(0) > th_assoc)
                    if aux_ids.shape[1] == 2:
                        sim_idxs = orig_ids[aux_ids[:,1]]
                    else:
                        sim_idxs = orig_ids[aux_ids.flatten()]
                    print('sim_idxs:',sim_idxs)
                    print('sim_idxs:',sim_idxs.shape)
                    print('mask shape:',mask.shape)
                    print('idxs mask:',torch.nonzero(mask))

                mask[:,torch.unique(sim_idxs)] = False

                track_mask  = torch.max(torch.sigmoid(track_queries['pred_logits_2_b']),2)[0] > 0.5 #probando
                track_queries['delete'][track_mask.clone().squeeze(0)] = 0
                track_queries['delete'][~track_mask.clone().squeeze(0)] += 1 #Probando
                #print('delete:',track_queries['delete'])

                #delete tracks
                idxs_keep = torch.nonzero(track_queries['delete'] < limit).flatten()
                track_queries['embeds_2'] = track_queries['embeds_2'][:,idxs_keep]
                track_queries['pred_logits_2'] = track_queries['pred_logits_2'][:,idxs_keep]
                track_queries['pred_boxes_2'] = track_queries['pred_boxes_2'][:,idxs_keep]
                track_queries['delete'] = track_queries['delete'][idxs_keep]
                track_queries['ids'] = track_queries['ids'][idxs_keep]

                track_queries['embeds_2'] = torch.cat((track_queries['embeds_2'],track_queries['hs'][mask.clone()].unsqueeze(0)),dim=1)
                track_queries['pred_logits_2'] = torch.cat((track_queries['pred_logits_2'],out_logits[-1][mask.clone()].unsqueeze(0)),dim=1)
                track_queries['pred_boxes_2'] = torch.cat((track_queries['pred_boxes_2'],out_bboxes[-1][mask.clone()].unsqueeze(0)),dim=1)
                track_queries['delete'] = torch.cat((track_queries['delete'],torch.zeros(track_queries['hs'][mask.clone()].shape[0]).to(track_queries['embeds_2'].device)),0)
                
                if track_queries['embeds_2'].shape[1] > 0:
                    if "id_max" not in track_queries:
                        track_queries['ids'] = torch.arange(0,track_queries['embeds_2'].shape[1]).to(track_queries['embeds_2'].device)
                    else:
                        id_max = track_queries['id_max']
                        track_queries["ids"] = torch.cat((track_queries["ids"],torch.arange(id_max+1,id_max+1+mask.sum()).to(track_queries["embeds_2"].device)))

                    track_queries['id_max'] = torch.max(track_queries["ids"])
        else:
            track_queries['re_boxes'] = None
            track_queries['re_logits'] = None
            if 'embeds_2' in track_queries:
                track_queries['delete'] += 1
                idxs_keep = torch.nonzero(track_queries['delete'] < limit).flatten()
                track_queries['embeds_2'] = track_queries['embeds_2'][:,idxs_keep]
                track_queries['pred_logits_2'] = track_queries['pred_logits_2'][:,idxs_keep]
                track_queries['pred_boxes_2'] = track_queries['pred_boxes_2'][:,idxs_keep]
                track_queries['delete'] = track_queries['delete'][idxs_keep]
                track_queries['ids'] = track_queries['ids'][idxs_keep]

                """
                _, boxes, scores, attn_weights = self.decoder_track(track_queries['embeds_2'],self.query_pos_head(track_queries['pred_boxes_2']),track_queries['pred_boxes_2'],track_queries['hs'][mask.clone()].unsqueeze(0),None,self.dec_bbox_head_track,self.dec_score_head_track)
                track_queries['mask'] = attn_weights
                if attn_weights.shape[0]>0:
                    track_queries['re_boxes'] = boxes
                    track_queries['re_logits'] = scores #revisar bien esta parte
                else:
                    track_queries['re_boxes'] = None
                    track_queries['re_logits'] = None
                """

        if 'embeds_2' in track_queries:
            if track_queries['embeds_2'].shape[1] == 0:
                track_queries.pop('embeds_2', None)
                track_queries.pop('pred_logits_2', None)
                track_queries.pop('pred_boxes_2', None)
                track_queries.pop('delete', None)
                track_queries.pop('ids', None)
                track_queries.pop('id_max', None)
            
        return out, track_queries
