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

from scipy.optimize import linear_sum_assignment


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
           self.cross_attn = MultiheadAttention_aux(d_model, n_head, dropout=dropout, batch_first=True)

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

        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        if self.deformable:
           tgt2, _ = self.cross_attn(\
                   self.with_pos_embed(tgt, query_pos_embed), 
                   reference_points, 
                   memory, 
                   memory_spatial_shapes, 
                   memory_mask)
        else:
           tgt2, attn_weights, attn_weights_logits = self.cross_attn(\
                   q,
                   memory,
                   memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        if return_attn:
           return tgt, attn_weights, attn_weights_logits
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
        ref_points_detach = F.sigmoid(ref_points_unact)
        sampling_values_list = []
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)
            output, sampling_values = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed) #add memory_mask
            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
            sampling_values_list.append(sampling_values)
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

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), output, sampling_values_list #track_queries


class QIM(nn.Module):
    def __init__(self,hidden_dim=256,dropout=0.2):
        super(QIM, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, 8, dropout,batch_first=True)
        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU(True)

    def forward(self,qdt,boxes,mask):
        #query_pos = pos2posemb(boxes)
        query_pos = boxes
        q = k = query_pos + qdt
        tgt = k = qdt
        tgt2, _ = self.self_attn(q,k,tgt,attn_mask=mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout0(self.activation(self.linear1(tgt))))
        #print('tgt2 shape:',tgt2.shape)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

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

        #contrastive
        self.proj_cons = MLP(hidden_dim, 2 * hidden_dim, hidden_dim, num_layers=6)
        #self.proj_cons = MLP(3072, 3072//2, hidden_dim, num_layers=6)
        #self.proj_boxes = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=6)

        #self.proj_samplings = MLP(3072, 2 * hidden_dim, hidden_dim, num_layers=6)

        #deformable track
        #decoder_layer_track = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)

        #interact queries
        #self.track_decoder_layer_h = TransformerDecoderLayerMOT_v2(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points,deformable=False)
        #self.track_decoder_layer_h = TransformerDecoderLayerMOT_v2(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points,deformable=False)
        """
        self.dec_score_head_track = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head_track = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_decoder_layers)
        ])
        """
        
        self.qim = QIM() #reid
        self.dec_score_head_track = nn.Linear(hidden_dim, num_classes) #reid
        self.dec_bbox_head_track = MLP(hidden_dim, hidden_dim, 4, 3) #reid
        

    def forward(self, feats, targets=None, track_queries=None, ind_track=True, criterion= None, mode='', limit=2, update_track=False): #limit=10 or 30 -> cuidado con el limite
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

        sampling_values_det = torch.stack(sampling_values_det[-1])
        #print('sampling_values_det:',sampling_values_det.shape)
        
        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            _ , track_queries['hs'] = torch.split(track_queries['hs'], dn_meta['dn_num_split'], dim=1)
            _, sampling_values_det = torch.split(sampling_values_det, dn_meta['dn_num_split'], dim=3)
        
        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        
        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        th = 0.5

        with torch.no_grad():
             #scores = F.sigmoid(out_logits[-1])
             #mask = torch.max(scores,2)[0] > th
             #Probando el guiar las predicciones con respecto a las gt
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
             #Probando para agregar mas ejemplos de instancias -- quitar si no mejora
             scores = F.sigmoid(out_logits[-1])
             mask = torch.max(scores,2)[0] > th
             track_queries['mask_pred'] = mask.clone()
             mask = mask.to('cuda')

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
            #track_queries=track_queries,
            mode=1)

            track_queries['embeds_2'] = output_2
            track_queries['pred_boxes_2'] = out_bboxes_2[-1]
            track_queries['pred_logits_2'] = out_logits_2[-1]

            track_queries['embeds_2_b'] = track_queries['embeds_2'].clone()
            track_queries['pred_logits_2_b'] = track_queries['pred_logits_2'].clone()
            track_queries['pred_boxes_2_b'] = track_queries['pred_boxes_2'].clone()

            sampling_values_track = torch.stack(sampling_values_track[-1])
        
        if mask.sum() > 0 and ind_track:
            #print('hs mask shape:',track_queries['hs'][mask.clone()].shape)
            #print('pred boxes mask shape:',self.proj_boxes(out_bboxes[-1][mask.clone()]).shape)
            track_queries['cons_det'] = self.proj_cons(track_queries['hs'][mask.clone()]) #sin utilizar coordendas
            #track_queries['cons_det'] = self.proj_cons(track_queries['hs'][mask.clone()]+self.proj_boxes(out_bboxes[-1][mask.clone()]))
            #track_queries['cons_det'] = self.proj_cons(torch.flatten(sampling_values_det[:,:,:,mask.clone().squeeze(0),:].permute(3,0,1,2,4),start_dim=1))
            #track_queries['cons_det'] = self.proj_cons(track_queries['hs'][mask.clone()]+self.proj_boxes(out_bboxes[-1][mask.clone()])+self.proj_samplings(torch.flatten(sampling_values_det[:,:,:,mask.clone().squeeze(0),:].permute(3,0,1,2,4),start_dim=1)))
            if 'embeds_2' not in track_queries:
                track_queries['embeds_2'] = track_queries['hs'][mask].unsqueeze(0)
                track_queries['pred_boxes_2'] = out_bboxes[-1][mask].unsqueeze(0)
                track_queries['pred_logits_2'] = out_logits[-1][mask].unsqueeze(0)
                track_queries['cons_track'] = self.proj_cons(track_queries['embeds_2'].squeeze(0)) #sin utilizar coordenadas
                #track_queries['cons_track'] = self.proj_cons(track_queries['embeds_2'].squeeze(0)+self.proj_boxes(track_queries['pred_boxes_2'].squeeze(0)))
                #track_queries['cons_track'] = self.proj_cons(torch.flatten(sampling_values_det[:,:,:,mask.clone().squeeze(0),:].permute(3,0,1,2,4),start_dim=1))
                #track_queries['cons_track'] = self.proj_cons(track_queries['embeds_2'].squeeze(0)+self.proj_boxes(track_queries['pred_boxes_2'].squeeze(0))+self.proj_samplings(torch.flatten(sampling_values_det[:,:,:,mask.clone().squeeze(0),:].permute(3,0,1,2,4),start_dim=1)))
                if "id_max" not in track_queries:
                   track_queries['ids'] = torch.arange(0,track_queries['embeds_2'].shape[1]).to(track_queries['embeds_2'].device)
                track_queries['id_max'] = torch.max(track_queries["ids"])
                track_queries['delete'] = torch.zeros(track_queries['embeds_2'].shape[1]).to(track_queries['embeds_2'].device)
            else:
                """
                #Update queries
                if update_track:
                    aux_hs = track_queries['hs'][mask.clone()]
                    aux_hs = aux_hs.unsqueeze(0)
                    aux_track_queries_tgt, attn_weights, attn_weights_logits = self.track_decoder_layer_h(track_queries["embeds_2"],None,aux_hs,None,None,None,None,self.query_pos_head(track_queries["pred_boxes_2"]), return_attn=True)
                    #aux_track_queries_tgt = self.track_decoder_layer_h(track_queries["embeds_2"],None,aux_hs,None,None,None,None,self.query_pos_head(track_queries["pred_boxes_2"]))
                    track_queries["embeds_2"] = aux_track_queries_tgt #update track queries by det queries
                """
                track_queries['cons_track'] = self.proj_cons(track_queries['embeds_2'].squeeze(0)) #sin utilizar coordenadas
                #track_queries['cons_track'] = self.proj_cons(track_queries['embeds_2'].squeeze(0)+self.proj_boxes(track_queries['pred_boxes_2'].squeeze(0)))
                #Etrack_queries['cons_track'] = self.proj_cons(torch.flatten(sampling_values_track.permute(3,0,1,2,4),start_dim=1))
                #track_queries['cons_track'] = self.proj_cons(track_queries['embeds_2'].squeeze(0)+self.proj_boxes(track_queries['pred_boxes_2'].squeeze(0))+self.proj_samplings(torch.flatten(sampling_values_track.permute(3,0,1,2,4),start_dim=1)))
                res = F.normalize(track_queries['cons_track']) @ F.normalize(track_queries['cons_det']).t()
                
                track_queries['embeds_2_b'] = track_queries['embeds_2'].clone()
                track_queries['pred_logits_2_b'] = track_queries['pred_logits_2'].clone()
                track_queries['pred_boxes_2_b'] = track_queries['pred_boxes_2'].clone()
                track_queries['ids_b'] = track_queries['ids'].clone()

                #th_assoc = 0.5
                th_assoc = 0.5 #0.5 train #0.3 test
                if self.training:
                    ious = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['pred_boxes_2'].squeeze(0)),box_cxcywh_to_xyxy(out_bboxes[-1][mask.clone()]))
                    sim_idxs = torch.nonzero(ious > th_assoc)[:,1]
                    sim_del = torch.nonzero(ious < th_assoc)[:,0]
                    sim_keep = torch.nonzero(ious > th_assoc)[:,0]
                else:
                    ious = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['pred_boxes_2'].squeeze(0)),box_cxcywh_to_xyxy(out_bboxes[-1][mask.clone()]))
                    sim_idxs = torch.nonzero(res > th_assoc)[:,1]
                    sim_del = torch.nonzero(res < th_assoc)[:,0]
                    sim_keep = torch.nonzero(res > th_assoc)[:,0]

                #Update track queries by associations
                #generate mask for transformer
                """
                if update_track:
                    aux_hs = track_queries['hs'][sim_idxs]
                    aux_hs = aux_hs.unsqueeze(0)
                    #print(sim_idxs.shape)
                    aux_track_queries_tgt, attn_weights, attn_weights_logits = self.track_decoder_layer_h(track_queries["embeds_2"],None,aux_hs,None,None,None,,self.query_pos_head(track_queries["pred_boxes_2"]), return_attn=True)
                    track_queries["embeds_2"] = aux_track_queries_tgt #update track queries by det queries
                """
                
                #Uso del modulo qim
                
                aux_queries = torch.cat((track_queries['embeds_2'],track_queries['hs'][mask.clone()].unsqueeze(0)),dim=1)
                aux_boxes = torch.cat((track_queries['pred_boxes_2'],out_bboxes[-1][mask.clone()].unsqueeze(0)),dim=1)
                if self.training:
                    ious_mask = generalized_box_iou(box_cxcywh_to_xyxy(aux_boxes.squeeze(0)),box_cxcywh_to_xyxy(aux_boxes.squeeze(0)))
                    #print('ious_mask:',ious_mask)
                else:
                    combine_cons = torch.cat((track_queries['cons_track'],track_queries['cons_det']),dim=0)
                    ious_mask = F.normalize(combine_cons) @ F.normalize(combine_cons).t()
                    cost_bbox = generalized_box_iou(box_cxcywh_to_xyxy(aux_boxes.squeeze(0)),box_cxcywh_to_xyxy(aux_boxes.squeeze(0)))
                    print('cost_bbox:',cost_bbox)
                    print('res:',ious_mask)

                ious_mask = ious_mask < 0.5
                print('ious_mask:',ious_mask)
                #track_queries['embeds_2_q'] = self.qim(aux_queries,self.query_pos_head(aux_boxes),ious_mask)
                track_queries['embeds_2_q'] = self.qim(aux_queries,self.query_pos_head(aux_boxes),ious_mask)
                #print('embeds_2_q shape:',track_queries['embeds_2_q'].shape)
                #print('split:',track_queries['embeds_2'].shape[1])
                #track_queries['embeds_2_q'], _ = torch.split(track_queries['embeds_2_q'], [track_queries['embeds_2'].shape[1],track_queries['hs'][mask.clone()].shape[0]], dim=1)
                print('tracks shape:',track_queries['embeds_2'].shape[1])
                print('shape dets:',track_queries['hs'][mask.clone()].shape[0])
                track_queries['embeds_2'], _ = torch.split(track_queries['embeds_2_q'], [track_queries['embeds_2'].shape[1],track_queries['hs'][mask.clone()].shape[0]], dim=1)
                #track_queries['boxes_re'] = F.sigmoid(self.dec_bbox_head_track(track_queries['embeds_2_q'])+inverse_sigmoid(track_queries['pred_boxes_2'].detach().clone())) #tener cuidado con este detach
                #track_queries['scores_re'] = self.dec_score_head_track(track_queries['embeds_2_q'])
                print('boxes antes:',track_queries['pred_boxes_2_b'])
                print('scores antes:',torch.sigmoid(track_queries['pred_logits_2_b']))
                track_queries['pred_boxes_2_b'] = F.sigmoid(self.dec_bbox_head_track(track_queries['embeds_2'])+inverse_sigmoid(track_queries['pred_boxes_2'].detach().clone())) #tener cuidado con este detach
                track_queries['pred_logits_2_b'] = self.dec_score_head_track(track_queries['embeds_2'])
                print('boxes despues:',track_queries['pred_boxes_2_b'])
                print('scores despues:',torch.sigmoid(track_queries['pred_logits_2_b']))
                #print('targets:',targets)
                
                
                mask[:,torch.unique(torch.nonzero(mask[0])[sim_idxs])] = False

                if not self.training:
                    track_mask  = torch.max(torch.sigmoid(track_queries['pred_logits_2_b']),2)[0] > 0.3 #probando
                    track_queries['delete'][track_mask.clone().squeeze(0)] = 0
                    track_queries['delete'][~track_mask.clone().squeeze(0)] += 1#Probando
                    #Utilizando la presencia de los detectores para la version 1
                    track_queries['delete'][torch.unique(sim_del)] += 1
                    track_queries['delete'][torch.unique(sim_keep)] = 0

                else:
                    #print('sim_keep:',sim_keep)
                    #print('sim_del:',sim_del)
                    track_queries['delete'][torch.unique(sim_del)] += 1
                    track_queries['delete'][torch.unique(sim_keep)] = 0
                """
                track_queries['delete'][torch.unique(sim_del)] += 1
                track_queries['delete'][torch.unique(sim_keep)] = 0
                """ #Volver a colocar  si lo de arriba no funciona

                #print('embeds_2 shape:',track_queries['embeds_2'].shape)

                #delete tracks
                idxs_keep = torch.nonzero(track_queries['delete'] < limit).flatten()
                idxs_delete = torch.nonzero(track_queries['delete'] >= limit).flatten()

                #Re-id
                if self.training:
                    if len(idxs_delete)>0 and 'ids_gt' in track_queries:
                        aux_ids = track_queries['ids'][idxs_delete]
                        ids_gt = [v for k,v in track_queries['ids_gt'].items() if v is not None]
                        ids_gt = torch.tensor(ids_gt).flatten().to('cuda:0')
                        aux_ids = [aux_id for aux_id in aux_ids if aux_id in ids_gt]
                        aux_idxs_delete = [i for i, x in enumerate(track_queries['ids']) if x in aux_ids]
                        if len(aux_idxs_delete)>0:
                            if 'history_embed' not in track_queries:
                                track_queries['history_embed'] = track_queries['embeds_2'][:,aux_idxs_delete].squeeze(0)
                                track_queries['history_id'] = track_queries['ids'][aux_idxs_delete]
                            else:
                                track_queries['history_embed'] = torch.cat((track_queries['history_embed'],track_queries['embeds_2'][:,aux_idxs_delete].squeeze(0)))
                                track_queries['history_id'] = torch.cat((track_queries['history_id'],track_queries['ids'][aux_idxs_delete]))
                else:
                    if len(idxs_delete)>0:
                        if 'history_embed' not in track_queries:
                            track_queries['history_embed'] = track_queries['embeds_2'][:,idxs_delete].squeeze(0)
                            track_queries['history_id'] = track_queries['ids'][idxs_delete]
                        else:
                            #print('history_embed shape:',track_queries['history_embed'].shape)
                            #print('embeds_2 shape:',track_queries['embeds_2'].shape)
                            track_queries['history_embed'] = torch.cat((track_queries['history_embed'],track_queries['embeds_2'][:,idxs_delete].squeeze(0)))
                            track_queries['history_id'] = torch.cat((track_queries['history_id'],track_queries['ids'][idxs_delete]))

                track_queries['embeds_2'] = track_queries['embeds_2'][:,idxs_keep]
                track_queries['pred_logits_2'] = track_queries['pred_logits_2'][:,idxs_keep]
                track_queries['pred_boxes_2'] = track_queries['pred_boxes_2'][:,idxs_keep]
                track_queries['delete'] = track_queries['delete'][idxs_keep]
                track_queries['ids'] = track_queries['ids'][idxs_keep]

                #Re-id
                #print('mask sum:',mask.sum())
                if 'history_embed' in track_queries and mask.sum()>0 and track_queries['history_embed'].shape[0]>0:
                    #print('history_embed shape:',track_queries['history_embed'].shape[0])
                    if self.training:
                        aux_outputs = {'pred_logits':out_logits[-1][mask].unsqueeze(0),'pred_boxes':out_bboxes[-1][mask].unsqueeze(0)}
                        indices = criterion.matcher(aux_outputs, targets)
                        ids_assoc = targets[0]['ids'][indices[0][1]].tolist()
                        if 'ids_gt' in track_queries:
                           ids_hist = torch.tensor([[i, torch.nonzero(track_queries['history_id']==track_queries['ids_gt'][ids_asoc]),track_queries['ids_gt'][ids_asoc]] for i,ids_asoc in enumerate(ids_assoc) if ids_asoc in track_queries['ids_gt'] and track_queries['ids_gt'][ids_asoc] is not None and track_queries['ids_gt'][ids_asoc] in track_queries['history_id'] ]).to('cuda:0')
                        else:
                           ids_hist = torch.tensor([])
                        if ids_hist.shape[0]>0:
                           idxs_det_new = torch.tensor([i for i in range(aux_outputs['pred_boxes'].shape[1]) if i not in ids_hist[:,0]])
                           track_queries['dict_ids'] = ids_hist
                           track_queries['hist_embed'] = self.proj_cons(track_queries['history_embed'])
                           track_queries['hist_embed_det'] = self.proj_cons(track_queries['hs'][:,ids_hist[:,0]]).squeeze(0)
                    else:
                        hist_embed = self.proj_cons(track_queries['history_embed'])
                        track_queries['cons_det'] = self.proj_cons(track_queries['hs'][mask])
                        print('cons_det shape:',track_queries['cons_det'].shape)
                        res_reid = F.normalize(track_queries['cons_det']) @ F.normalize(hist_embed).t()
                        print('res_reid:',res_reid)
                        v, idxs_hist = torch.max(res_reid,1)
                        aux_idxs = torch.nonzero(v>0.7)
                        print('aux_idxs shape:',aux_idxs.shape)
                        aux_re_id = res_reid[aux_idxs]
                        print('aux_re_id shape:',aux_re_id.shape)
                        if len(aux_re_id.shape) < 2:
                            aux_re_id = aux_re_id.unsqueeze(0)
                        if len(aux_re_id.shape) > 2:
                            aux_re_id = aux_re_id.squeeze(0)
                        print('aux_re_id:',aux_re_id)
                        print('aux_re_id shape:',aux_re_id.shape)
                        if len(aux_re_id.shape) > 2:
                            aux_re_id = aux_re_id.squeeze(1)
                        print('aux_re_id shape:',aux_re_id.shape)
                        if aux_re_id.shape[0] > 0:
                            indices = linear_sum_assignment(aux_re_id.cpu().numpy())
                            indices = np.asarray(indices)
                            indices = np.transpose(indices)
                            indices = torch.tensor(indices).to('cuda:0')
                            print('indices shape:',indices.shape)
                            print('indices:',indices)
                            print('history_id shape:',track_queries['history_id'][indices[:,1]].shape)
                            print('history_id:',track_queries['history_id'][indices[:,1]])
                            ids_hist = torch.cat((indices,track_queries['history_id'][indices[:,1]].unsqueeze(0).t()),dim=1)
                            print('ids_hist:',ids_hist)
                            print('ids_hist shape:',ids_hist.shape)
                            idxs_det_new = torch.tensor([i for i in range(track_queries['cons_det'].shape[0]) if i not in ids_hist[:,0]])
                            print('idxs_det_new:',idxs_det_new)
                        else:
                            ids_hist = None
                            idxs_det_new = None
                        #idxs_det_new = idxs_det_new.squeeze(1)

                #Re-id
                #track_queries['embeds_2_q'] = track_queries['embeds_2_q'][:,idxs_keep]

                #Probando el actualizar los tracks con las detecciones #reid
                #track_queries['embeds_2'] = track_queries['embeds_2_q'] #probar y tener cuidado

                track_queries['embeds_2'] = torch.cat((track_queries['embeds_2'],track_queries['hs'][mask.clone()].unsqueeze(0)),dim=1)
                track_queries['pred_logits_2'] = torch.cat((track_queries['pred_logits_2'],out_logits[-1][mask.clone()].unsqueeze(0)),dim=1)
                track_queries['pred_boxes_2'] = torch.cat((track_queries['pred_boxes_2'],out_bboxes[-1][mask.clone()].unsqueeze(0)),dim=1)
                track_queries['delete'] = torch.cat((track_queries['delete'],torch.zeros(track_queries['hs'][mask.clone()].shape[0]).to(track_queries['embeds_2'].device)),0)

                if track_queries['embeds_2'].shape[1] > 0:
                    if "id_max" not in track_queries:
                        track_queries['ids'] = torch.arange(0,track_queries['embeds_2'].shape[1]).to(track_queries['embeds_2'].device)
                    else:
                        #Re-id
                        id_max = track_queries['id_max']
                        if 'history_embed' in track_queries and track_queries['history_embed'].shape[0]>0 and mask.sum() > 0 and ids_hist is not None and len(ids_hist)>0:
                            aux_delete_ids = [x for x,z in enumerate(track_queries['history_id']) if x not in ids_hist[:,1]]
                            track_queries['history_embed'] = track_queries['history_embed'][aux_delete_ids]
                            track_queries['history_id'] = track_queries['history_id'][aux_delete_ids]
                            aux_ids = torch.zeros(mask.sum(),dtype=torch.long).to(track_queries["embeds_2"].device)
                            aux_ids[ids_hist[:,0]] = ids_hist[:,2]
                            if len(idxs_det_new)>0:
                                aux_ids[idxs_det_new] = torch.arange(id_max+1,id_max+1+len(idxs_det_new)).to(track_queries["embeds_2"].device)
                            track_queries['ids'] = torch.cat((track_queries['ids'],aux_ids))
                        else:
                            track_queries['ids'] = torch.cat((track_queries['ids'],torch.arange(id_max+1,id_max+1+mask.sum()).to(track_queries["embeds_2"].device)))
                    if 'history_id_max' not in track_queries:
                        track_queries['history_id_max'] = torch.unique(track_queries['ids'])
                    else:
                        track_queries['history_id_max'] = torch.unique(torch.cat((track_queries['history_id_max'],track_queries['ids'])))
                    track_queries['id_max'] = torch.max(track_queries['history_id_max'])

        #add delete by score
        if 'embeds_2' in track_queries:
            aux_track_mask  = torch.max(torch.sigmoid(track_queries['pred_logits_2']),2)[0] > 0.5
            aux_track_mask = aux_track_mask.squeeze(0)
            track_queries['delete'][(aux_track_mask == 0).nonzero().flatten()] += 1
            track_queries['delete'][aux_track_mask] = 0

            idxs_keep = torch.nonzero(track_queries['delete'] < limit).flatten()
            #re-id
            if ind_track:
               idxs_delete = torch.nonzero(track_queries['delete'] >= limit).flatten()
               if len(idxs_delete)>0:

                  if 'history_embed' not in track_queries:
                      track_queries['history_embed'] = track_queries['embeds_2'][:,idxs_delete].squeeze(0)
                      track_queries['history_id'] = track_queries['ids'][idxs_delete]
                  else:
                      track_queries['history_embed'] = torch.cat((track_queries['history_embed'],track_queries['embeds_2'][:,idxs_delete].squeeze(0)))
                      track_queries['history_id'] = torch.cat((track_queries['history_id'],track_queries['ids'][idxs_delete]))
            
            track_queries['embeds_2'] = track_queries['embeds_2'][:,idxs_keep]
            track_queries['pred_logits_2'] = track_queries['pred_logits_2'][:,idxs_keep]
            track_queries['pred_boxes_2'] = track_queries['pred_boxes_2'][:,idxs_keep]
            track_queries['delete'] = track_queries['delete'][idxs_keep]
            track_queries['ids'] = track_queries['ids'][idxs_keep]

        return out, track_queries
