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
import os

import matplotlib.pyplot as plt


__all__ = ['RTDETRTransformerMOT_v2']


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)


        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q):
        k = self.k_linear(q)
        q = self.q_linear(q)
        #k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        #qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        #kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.bmm(q * self.normalize_fact, k.transpose(-2, -1))
        #weights = self.dropout(weights)
        return weights


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
            #value_mask = value_mask.astype(value.dtype).unsqueeze(-1) #quitar si la modificacion de abajo no funciona
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
        #new

        aux_shapes = torch.tensor(value_spatial_shapes, device=value.device)
        sampling_coords = sampling_locations.clone().detach()
        sampling_coords = sampling_coords * aux_shapes[None,None,None,:,None,:]
        sampling_coords = sampling_coords.to(torch.int64)

        output, sampling_values = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        attn_weights = output.clone().detach()

        output = self.output_proj(output)

        return output, attn_weights, sampling_values, sampling_coords, attention_weights

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

        #self_attn
        self.self_attn = MultiheadAttention_aux(d_model, n_head, dropout=dropout, batch_first=True)

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
                cross=True):
        # self attention
        #q = k = self.with_pos_embed(tgt, query_pos_embed)
        # if attn_mask is not None:
        #     attn_mask = torch.where(
        #         attn_mask.to(torch.bool),
        #         torch.zeros_like(attn_mask),
        #         torch.full_like(attn_mask, float('-inf'), dtype=tgt.dtype))
        """v8
        # cross attention
        tgt2, sampling_values = self.cross_attn(\
            self.with_pos_embed(tgt, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
                
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        tgt_0 = tgt.clone()

        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        """
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt2, attn_scores, attn_scores_logits = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2, attn_weights_sampling, sampling_values, sampling_locations, attention_weights = self.cross_attn(\
            self.with_pos_embed(tgt, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)        

        #return tgt_0, tgt, attention_scores
        return None, tgt, attn_scores_logits, attn_weights_sampling, sampling_values, sampling_locations, attention_weights

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
                mask_head,
                num_denoising,
                attn_mask=None,
                memory_mask=None,
                track_queries=None,
                return_attn=True,
                mode=0):

        output = tgt

        dec_out_bboxes = []
        dec_out_bboxes_0 = []
        dec_out_logits = []
        dec_out_logits_0 = []
        attention_scores = []
        sampling_values = []
        sampling_locations = []
        attention_weights = []
        attn_weights_samplings = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)
            #attn_mask_aux = attn_mask.clone()
            #print('attn mask sahpe:',attn_mask_aux.shape)
            #print('num denoising:',num_denoising)
            #print('output shape:',output.shape)
            #attn_mask_aux[num_denoising:,num_denoising:] = mask_head[i](output[:,num_denoising:]) #update attnmask with learn weights
            #print('attn_mask shape before:',attn_mask.shape)
            attn_mask_aux = attn_mask.clone()
            attn_mask_aux[num_denoising:,num_denoising:] = mask_head[i](output[:,num_denoising:]) #update attnmask with learn weights
            #print('attn mask shape:',attn_mask.shape)
            attention_scores.append(attn_mask_aux)
            #attn_mask = attn_mask_aux #update attnmask with learn weights
            output_0, output, attn_score, attn_weights_sampling, sampling_value, sampling_location, attn_weights = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask_aux, memory_mask, query_pos_embed) #add memory_mask
                           #attn_mask, memory_mask, query_pos_embed) #add memory_mask
            #attention_scores.append(attn_score)
            #sampling_values.append(sampling_value)
            #sampling_locations.append(sampling_location)
            #attention_weights.append(attn_weights)
            #attn_weights_samplings.append(attn_weights_sampling)
            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                #dec_out_logits_0.append(score_head[i](output_0))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                    #dec_out_bboxes_0.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))
                    #dec_out_bboxes_0.append(F.sigmoid(bbox_head[i](output_0) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                #dec_out_logits_0.append(score_head[i](output_0))
                dec_out_bboxes.append(inter_ref_bbox)
                #dec_out_bboxes_0.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox
        #return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), output, torch.stack(attention_scores), torch.stack(dec_out_bboxes_0), torch.stack(dec_out_logits_0)
        #return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), output, torch.stack(attention_scores), attn_weights_samplings, sampling_values, torch.stack(sampling_locations), attention_weights, None, None
        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), output, torch.stack(attention_scores), None, None, None, None, None, None

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

        decoder_layer = TransformerDecoderLayerMOT_v2(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        self.decoder = TransformerDecoderMOT_v2(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        self.masking = nn.ModuleList([
            MHAttentionMap(hidden_dim,hidden_dim,nhead)
            for _ in range(num_decoder_layers)
        ])

    def forward(self, feats, targets=None, track_queries=None, ind_track=True, criterion= None, mode='', limit=2, update_track=False): #limit=10 or 30 -> cuidado con el limite
        # input projection and embedding
        ind_track = False
        device = feats[0].device
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

        aux_target = target.clone().detach()
        #print('orig target shape:',target.shape)
        if 'embeds' in track_queries and track_queries['embeds'] is not None and ind_track:
            #Cambio en la mascara de atencion
            target = torch.cat((target,track_queries['embeds']),dim=1)
            init_ref_points_unact = torch.cat((init_ref_points_unact,inverse_sigmoid(track_queries['boxes'])),dim=1)
            if attn_mask is None:
               attn_mask = torch.full([track_queries['embeds'].shape[1]+aux_target.shape[1], track_queries['embeds'].shape[1]+aux_target.shape[1]], False, dtype=torch.bool, device=device)
               attn_mask[:aux_target.shape[1],aux_target.shape[1]:] = True 
               attn_mask[aux_target.shape[1]:,:aux_target.shape[1]] = True
            else:
               attn_mask_0 = torch.full((aux_target.shape[1],track_queries['embeds'].shape[1]),True,dtype=torch.bool, device=device)
               attn_mask_1 = torch.full((track_queries['embeds'].shape[1],aux_target.shape[1]+track_queries['embeds'].shape[1]),True,dtype=torch.bool, device=device)
               attn_mask_1[:,aux_target.shape[1]:]=False
               attn_mask = torch.cat((attn_mask,attn_mask_0),dim=1)
               attn_mask = torch.cat((attn_mask,attn_mask_1),dim=0)

        if attn_mask is not None:
           attn_mask = attn_mask.float() #convert bool mask to float
           attn_mask[attn_mask>0] = float("-inf")
        else:
            attn_mask = torch.zeros((self.num_queries,self.num_queries)).to(device)

        if dn_meta is not None:
            num_denoising_aux = dn_meta['dn_num_split'][0]
        else:
            num_denoising_aux = 0
        # decoder
        out_bboxes, out_logits, output, attention_scores, attn_weights_samplings, sampling_values, sampling_locations, attention_weights, out_bboxes_0, out_logits_0 = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.masking,
            #self.num_denoising,
            num_denoising_aux,
            attn_mask=attn_mask,
            return_attn=True,
            mode=0)
        attention_scores = torch.sigmoid(attention_scores[-1].squeeze(0))
        """
        #Revisar esta parte ya que estos scores se pueden usar de otra forma
        attention_scores = torch.mean(attention_scores,0)
        attention_scores = torch.mean(attention_scores,0)
        #attention_scores = F.normalize(attention_scores)
        attention_scores = torch.sigmoid(attention_scores)
        attention_scores_r = torch.relu(attention_scores)
        #print(torch.min(attention_scores))
        #print(torch.max(attention_scores))
        """
        t_num_split = 0
        if 'embeds' in track_queries and track_queries['embeds'] is not None and ind_track:
           t_num_split = track_queries['embeds'].shape[1]

        if self.training and dn_meta is not None:
            if t_num_split > 0:
               dn_meta['dn_num_split'][1]+=t_num_split
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            #_, sampling_locations = torch.split(sampling_locations, dn_meta['dn_num_split'], dim=2)
            _ , output = torch.split(output, dn_meta['dn_num_split'], dim=1)
            _, attention_scores = torch.split(attention_scores, dn_meta['dn_num_split'], dim=0)
            _, attention_scores = torch.split(attention_scores, dn_meta['dn_num_split'], dim=1)

        if 'embeds' in track_queries and track_queries['embeds'] is not None and ind_track:
            t_num_split = track_queries['embeds'].shape[1]
            det_num_split = out_bboxes.shape[2]-t_num_split
            output, track_queries['embeds'] = torch.split(output,[det_num_split,t_num_split],dim=1)
            out_bboxes, track_queries['boxes'] = torch.split(out_bboxes,[det_num_split,t_num_split],dim=2)
            track_queries['aux_boxes'] = track_queries['boxes']
            track_queries['boxes'] = track_queries['aux_boxes'][-1]
            out_logits, track_queries['logits'] = torch.split(out_logits,[det_num_split,t_num_split],dim=2)
            track_queries['aux_logits'] = track_queries['logits']
            track_queries['logits'] = track_queries['aux_logits'][-1]
            sampling_locations, track_queries['locations'] = torch.split(sampling_locations,[det_num_split,t_num_split],dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1],'attention_scores':attention_scores}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        th = 0.5
        scores = F.sigmoid(out_logits[-1])
        mask = torch.max(scores,2)[0] > th
        mask = mask.detach()

        #Probando visualizar:
        if not self.training and mask.sum() > 0:
           os.makedirs('vis_attn',exist_ok=True)
           idxs = torch.nonzero(mask)[:,1]
           boxes_det = out_bboxes[-1][0,idxs]

        """
        if not self.training and mask.sum() > 0:
           os.makedirs('vis_attn',exist_ok=True)
           #print('targets:',targets)
           lvl = 0
           head = 0
           h, w = 640, 640
           h_s, w_s = spatial_shapes[lvl]
           #print('hs,ws:',h_s,w_s)
           heads = attention_weights[-1].shape[2]
           idxs = torch.nonzero(mask)[:,1]
           #print('boxes shape:',out_bboxes.shape)
           aux_shapes = torch.tensor([[w/w_s,h/h_s]]).to(device)
           for j, idx in enumerate(idxs):
               for head in range(heads):
                  #weights = attention_weights[-1][0,mask.flatten(),head,lvl] #q0,h0,lvl0 -> 4
                  locations = sampling_locations[-1][0,idx,head,lvl] # -> 4,2
                  box = out_bboxes[-1][0,idx] # -> 4,2
                  #print('box shape:',box.shape)
                  #print('box antes:',box)
                  box = box_cxcywh_to_xyxy(box)
                  box *= torch.tensor([w,h,w,h]).to(device)
                  #print('box despues:',box)
                  locations = locations * aux_shapes[:,None,:]
                  locations = locations.to(torch.int64)
                  #print('locations shape:',locations)
                  img_mask = torch.zeros((640,640))
                  img_mask[locations[:,0],locations[:,1]] = 1
                  #print('img_mak sum:',img_mask.sum())
                  img_mask = img_mask.detach().cpu().numpy()
                  aux_mask = img_mask * 255
                  cv2.imwrite(os.path.join('vis_attn','box_'+str(j)+'_head_'+str(head)+'.jpg'),aux_mask)
                  plt_mask = plt.imshow(aux_mask) 
                  plt.savefig(os.path.join('vis_attn','box_'+str(j)+'_head_map_'+str(head)+'.jpg'))
                  aux_mask = np.repeat(aux_mask[:, :, np.newaxis], 3, axis=2)
                  #print('box int:',int(box[0]),int(box[1]),int(box[2]),int(box[3]))
                  cv2.rectangle(aux_mask, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255,0,0), (1))
                  cv2.imwrite(os.path.join('vis_attn','box_rect_'+str(j)+'_head_'+str(head)+'.jpg'),aux_mask)
        """
        """
        if not self.training:
            print('boxes shape:',out_bboxes.shape)
            ious = generalized_box_iou(box_cxcywh_to_xyxy(out_bboxes.squeeze(0).squeeze(0)),box_cxcywh_to_xyxy(out_bboxes.squeeze(0).squeeze(0)))
            print('ious:',ious.shape)
            print('attentions:',attention_scores.shape)
            print('box 0:',ious[0])
            print('attn 0:',attention_scores[0])
        """

        if not self.training:
           cost_bbox = generalized_box_iou(box_cxcywh_to_xyxy(out['pred_boxes'].squeeze(0)), box_cxcywh_to_xyxy(out['pred_boxes'].squeeze(0)))
           print('cost_bbox:',cost_bbox[mask.squeeze(0)])
           print('attentions:',attention_scores[mask.squeeze(0)])
           #print('attentions:',attention_scores[mask.squeeze(0)])
           #print('attn relu:',attention_scores_r)
        
        if criterion is not None:
            losses_dict, track_queries = criterion.match(out, track_queries, targets)

        """
        if mask.sum() > 0 and ind_track:
            if 'embeds' not in track_queries:
                track_queries['embeds'] = output[mask].unsqueeze(0)
                track_queries['aux_boxes'] = out_bboxes[:,mask].unsqueeze(1)
                track_queries['aux_logits'] = out_logits[:,mask].unsqueeze(1)
                track_queries['boxes'] = out_bboxes[-1][mask].unsqueeze(0)
                track_queries['logits'] = out_logits[-1][mask].unsqueeze(0)
                track_queries['ids'] = torch.zeros(out_logits[-1][mask].shape[0])
            else:
                print('pasa sum')
                if self.training:
                  cost_bbox = generalized_box_iou(box_cxcywh_to_xyxy(out['pred_boxes'].squeeze(0)), box_cxcywh_to_xyxy(track_queries['boxes'].squeeze(0)))
                  max_values = torch.max(cost_bbox[mask.squeeze(0)],dim=1)[0]
                else:
                  aux_attn = attention_scores[:out['pred_boxes'].shape[1],:]
                  max_values = torch.max(aux_attn[mask.squeeze(0)],dim=1)[0]
                  if torch.nonzero(max_values > th).shape[0]>0:
                     print('aquii idxs_det:',torch.nonzero(max_values > th))

                mask_values = max_values < th
                mask_values = mask_values.detach()
                
                track_queries['embeds'] = torch.cat((track_queries['embeds'],output[mask][mask_values].unsqueeze(0)),dim=1)
                track_queries['aux_boxes'] = torch.cat((track_queries['aux_boxes'],out_bboxes[:,mask][:,mask_values].unsqueeze(1)),dim=2)
                track_queries['aux_logits'] = torch.cat((track_queries['aux_logits'],out_logits[:,mask][:,mask_values].unsqueeze(1)),dim=2)
                track_queries['boxes'] = track_queries['aux_boxes'][-1]
                track_queries['logits'] = track_queries['aux_logits'][-1]
                track_queries['ids'] = torch.cat((track_queries['ids'],torch.zeros(output[mask][mask_values].shape[0])))
        """
        if self.training:
            return losses_dict, track_queries
        else:
            return out, track_queries