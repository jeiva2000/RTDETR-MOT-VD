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
                 aux_loss=True,
                 ):

        super().__init__(num_classes,hidden_dim,num_queries,position_embed_type,
                         feat_channels,feat_strides,num_levels,num_decoder_points,
                         nhead,num_decoder_layers,dim_feedforward,dropout,activation,
                         num_denoising,label_noise_ratio,box_noise_scale,learnt_init_query,
                         eval_spatial_size,eval_idx,eps,aux_loss)
        self.mode = 3
        self.use_qim = True
        self.use_reid = False #True
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        self.decoder = TransformerDecoderMOT_v2(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        #contrastive
        if self.mode == 0:
            self.proj_cons = MLP(hidden_dim, 2 * hidden_dim, hidden_dim, num_layers=6)
        if self.mode == 1:
            self.proj_cons = MLP(hidden_dim, 2 * hidden_dim, hidden_dim, num_layers=6)
            self.proj_boxes = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=6)
        if self.mode == 2:
            self.proj_cons = MLP(3072, 3072//2, hidden_dim, num_layers=6)
        if self.mode == 3:
            self.proj_cons = MLP(hidden_dim, 2 * hidden_dim, hidden_dim, num_layers=6)
            self.proj_boxes = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=6)
            self.proj_samplings = MLP(3072, 2 * hidden_dim, hidden_dim, num_layers=6)

        """
        #opcional para probar
        if self.mode == 0:
            self.proj_cons_d = MLP(hidden_dim, 2 * hidden_dim, hidden_dim, num_layers=6)
        if self.mode == 1:
            self.proj_cons_d = MLP(hidden_dim, 2 * hidden_dim, hidden_dim, num_layers=6)
            self.proj_boxes_d = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=6)
        if self.mode == 2:
            self.proj_cons_d = MLP(3072, 3072//2, hidden_dim, num_layers=6)
        if self.mode == 3:
            self.proj_cons_d = MLP(hidden_dim, 2 * hidden_dim, hidden_dim, num_layers=6)
            self.proj_boxes_d = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=6)
            self.proj_samplings_d = MLP(3072, 2 * hidden_dim, hidden_dim, num_layers=6)
        """

        if self.use_qim:
            self.qim = QIM()
            self.dec_score_head_track = nn.Linear(hidden_dim, num_classes) 
            self.dec_bbox_head_track = MLP(hidden_dim, hidden_dim, 4, 3)

        #Ids
        #self.ids_head = MLP(hidden_dim,2 * hidden_dim, 300, num_layers=3)

    def forward(self, feats, targets=None, track_queries=None, ind_track=True, criterion= None, mode='', limit=3, update_track=False): #limit=10 or 30 -> cuidado con el limite
        # input projection and embedding
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
        if 'embeds' in track_queries and track_queries['embeds'] is not None:
            """
            target = torch.cat((track_queries['embeds'],target),dim=1)
            init_ref_points_unact = torch.cat((inverse_sigmoid(track_queries['boxes']),init_ref_points_unact),dim=1)
            #update mask --revisaar
            attn_mask = torch.full([track_queries['embeds'].shape[1]+aux_target.shape[1], track_queries['embeds'].shape[1]+aux_target.shape[1]], False, dtype=torch.bool, device=device)
            attn_mask[:track_queries['embeds'].shape[1],track_queries['embeds'].shape[1]:] = True 
            attn_mask[track_queries['embeds'].shape[1]:,:track_queries['embeds'].shape[1]] = True
            """
            """
            target = torch.cat((target,track_queries['embeds']),dim=1)
            init_ref_points_unact = torch.cat((init_ref_points_unact,inverse_sigmoid(track_queries['boxes'])),dim=1)
            #update mask --revisaar
            attn_mask = torch.full([track_queries['embeds'].shape[1]+aux_target.shape[1], track_queries['embeds'].shape[1]+aux_target.shape[1]], False, dtype=torch.bool, device=device)
            attn_mask[:aux_target.shape[1],aux_target.shape[1]:] = True 
            attn_mask[aux_target.shape[1]:,:aux_target.shape[1]] = True
            """
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
            mode=0)

        if self.mode in [2,3]:
            sampling_values_det = torch.stack(sampling_values_det[-1])

        t_num_split = 0
        if 'embeds' in track_queries and track_queries['embeds'] is not None and ind_track:
            t_num_split = track_queries['embeds'].shape[1]

        if self.training and dn_meta is not None:
            dn_meta['dn_num_split'][1]+=t_num_split
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            _ , output = torch.split(output, dn_meta['dn_num_split'], dim=1)
            if self.mode in [2,3]:
                _, sampling_values_det = torch.split(sampling_values_det, dn_meta['dn_num_split'], dim=3)

        if 'embeds' in track_queries and track_queries['embeds'] is not None:
            det_num_split = out_bboxes.shape[2]-t_num_split
            """
            if not self.training:
                print('det_num_split:',det_num_split)
                print('t_num_split:',t_num_split)
                print('output shape:',output.shape)
            """
            output, track_queries['embeds'] = torch.split(output,[det_num_split,t_num_split],dim=1)
            out_bboxes, track_queries['boxes'] = torch.split(out_bboxes,[det_num_split,t_num_split],dim=2)
            track_queries['boxes'] = track_queries['boxes'][-1]
            out_logits, track_queries['logits'] = torch.split(out_logits,[det_num_split,t_num_split],dim=2)
            track_queries['logits'] = track_queries['logits'][-1]
            """
            if not self.training:
                print('scores despues decoder:',torch.sigmoid(track_queries['logits']))
            """
            if self.mode in [2,3]:
                sampling_values_det, sampling_values_track = torch.split(sampling_values_det, [det_num_split,t_num_split], dim=3)
        
        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        
        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        th = 0.5
        
        if self.use_reid and self.training:
            ious = generalized_box_iou(box_cxcywh_to_xyxy(out_bboxes[-1].squeeze(0)),box_cxcywh_to_xyxy(targets[0]['boxes']))
            #print('ious:',ious)
            if ious.shape[1]>0:
                values, indices_b = torch.max(ious,1)
                mask = values > th
            else:
                mask = torch.zeros(out_logits[-1].shape[1])
                mask = mask.to(device)
            mask = mask.detach()
            track_queries['mask_pred'] = mask.clone()
        else:
            scores = F.sigmoid(out_logits[-1])
            mask = torch.max(scores,2)[0] > th
            mask = mask.detach()
            track_queries['mask_pred'] = mask.clone()
        

        if mask.sum() > 0 and ind_track:

            #reid
            #Probando para evitar los duplicados en el reid #probar quitando
            """
            if self.use_reid:
                aux_idxs_mask = torch.nonzero(mask.clone())[:,1]
                if self.training:
                    ious_hist = generalized_box_iou(box_cxcywh_to_xyxy(out_bboxes[-1][mask.clone()]),box_cxcywh_to_xyxy(out_bboxes[-1][mask.clone()]))
                else:
                    if self.mode==0:
                        aux_cons_det = self.proj_cons(output[mask.clone()])
                    if self.mode==1:
                        aux_cons_det = self.proj_cons(output[mask.clone()]+self.proj_boxes(out_bboxes[-1][mask.clone()]))
                    if self.mode==2:
                        aux_cons_det = self.proj_cons(torch.flatten(sampling_values_det[:,:,:,mask.clone().squeeze(0),:].permute(3,0,1,2,4),start_dim=1))
                    if self.mode==3:
                        aux_cons_det = self.proj_cons(output[mask.clone()]+self.proj_boxes(out_bboxes[-1][mask.clone()])+self.proj_samplings(torch.flatten(sampling_values_det[:,:,:,mask.clone().squeeze(0),:].permute(3,0,1,2,4),start_dim=1)))
                    ious_hist = F.normalize(aux_cons_det) @ F.normalize(aux_cons_det).t()
                indexs_0 = []
                indexs = []
                with torch.no_grad():
                    aux_scores = torch.max(torch.sigmoid(out_logits[-1][mask.clone()]),1)[0]
                    n_aux_scores,index_s = torch.sort(aux_scores,-1,descending=True)
                    index_s = index_s.detach().cpu().numpy()
                    for j in index_s:
                        aux = ious_hist[j]
                        if j not in indexs:
                            indexs_0.append(j)
                            aux_idxs_n = torch.where(torch.nonzero(aux>0.5).flatten()!=j)[0]
                            aux_idxs_n = aux_idxs_n.detach().cpu().numpy()
                            if len(aux_idxs_n)>0:
                                indexs.extend(aux_idxs_n)
                    indexs = np.array(indexs)
                    indexs = np.unique(indexs)
                
                if len(indexs)>0:
                    indexs = torch.tensor(indexs)
                    indexs = indexs.to(device)
                    mask[:,aux_idxs_mask[indexs]] = False
            """
                
            if 'embeds' not in track_queries or track_queries['embeds'].shape[1] == 0:
                track_queries['embeds'] = output[mask].unsqueeze(0)
                track_queries['boxes'] = out_bboxes[-1][mask].unsqueeze(0)
                track_queries['logits'] = out_logits[-1][mask].unsqueeze(0)
                if self.mode in [2,3]:
                    sampling_values_track = sampling_values_det[:,:,:,mask.squeeze(0),:]

                if "id_max" in track_queries:
                    id_max = track_queries['id_max']
                    aux_ids = torch.arange(id_max+1,id_max+1+mask.sum()).to(device)
                    print('ids antes:',track_queries['ids'])
                    track_queries['ids'] = torch.cat((track_queries['ids'],aux_ids))
                    print('ids despues:',track_queries['ids'])
                else:
                    track_queries['ids'] = torch.arange(0,mask.sum()).to(device)
                    print('ids despues:',track_queries['ids'])
                
                if 'history_id_max' not in track_queries:
                    track_queries['history_id_max'] = torch.unique(track_queries['ids'])
                else:
                    track_queries['history_id_max'] = torch.unique(torch.cat((track_queries['history_id_max'],track_queries['ids'])))

                track_queries['id_max'] = torch.max(track_queries['history_id_max'])
                track_queries['delete'] = torch.zeros(track_queries['embeds'].shape[1]).to(device)

                #Add gt_ids
                if self.use_reid and self.training:
                    ious_hist = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['boxes'].squeeze(0)),box_cxcywh_to_xyxy(targets[0]['boxes']))
                    v, aux_idxs = torch.max(ious_hist,1)
                    aux_idxs = aux_idxs[v>0.5]
                    #print('aux_idxs:',aux_idxs)
                    #print('gt ids:',targets[0]['ids'])
                    track_queries['gt_ids'] = targets[0]['ids'][aux_idxs]
                    #print('embeds shape:',track_queries['embeds'].shape)
                    #print('gt_ids shape:',track_queries['embeds'].shape)
                """
                if not self.training:
                    print('agrega primeros track y retorna')
                    print('embeds shape:',track_queries['embeds'].shape)
                    print('boxes:',track_queries['boxes'].shape)
                    print('scores:',torch.sigmoid(track_queries['logits']))
                    print('scores:',track_queries['logits'].shape)
                    print('ids:',track_queries['ids'].shape)
                    print('delete:',track_queries['delete'].shape)
                """
                if criterion is not None:
                    losses_dict, track_queries = criterion.match(out, track_queries, targets)

                track_queries['cons_det'] = None
                if self.training:
                    return losses_dict, track_queries
                else:
                    return out, track_queries 

            #Generate projections
            if self.mode==0:
                track_queries['cons_det'] = self.proj_cons(output[mask])
            if self.mode==1:
                track_queries['cons_det'] = self.proj_cons(output[mask]+self.proj_boxes(out_bboxes[-1][mask]))
            if self.mode==2:
                track_queries['cons_det'] = self.proj_cons(torch.flatten(sampling_values_det[:,:,:,mask.squeeze(0),:].permute(3,0,1,2,4),start_dim=1))
            if self.mode==3:
                track_queries['cons_det'] = self.proj_cons(output[mask]+self.proj_boxes(out_bboxes[-1][mask])+self.proj_samplings(torch.flatten(sampling_values_det[:,:,:,mask.squeeze(0),:].permute(3,0,1,2,4),start_dim=1)))
            """
            if not self.training:
                print('genera proyeccion de dets')
            """

        #if 'embeds' in track_queries and track_queries['embeds'].shape[0]>0 and mask.sum() > 0:
        if 'embeds' in track_queries and track_queries['embeds'].shape[0]>0:
            if self.mode==0:
                track_queries['cons_track'] = self.proj_cons(track_queries['embeds'].squeeze(0))  
            if self.mode==1:
                track_queries['cons_track'] = self.proj_cons(track_queries['embeds'].squeeze(0)+self.proj_boxes(track_queries['boxes'].squeeze(0)))
            if self.mode==2:
                track_queries['cons_track'] = self.proj_cons(torch.flatten(sampling_values_track.permute(3,0,1,2,4),start_dim=1))
            if self.mode==3:
                track_queries['cons_track'] = self.proj_cons(track_queries['embeds'].squeeze(0)+self.proj_boxes(track_queries['boxes'].squeeze(0))+self.proj_samplings(torch.flatten(sampling_values_track.permute(3,0,1,2,4),start_dim=1)))
            """
            if not self.training:
                print('genera cons_track')
            """
        """
        if 'cons_track' in track_queries and 'cons_det' in track_queries:
            res = F.normalize(track_queries['cons_track']) @ F.normalize(track_queries['cons_det']).t()
            if not self.training:
               print('genera res')
               print('res:',res)
        """

        th_assoc = 0.5 #0.5 train #0.3 test
        #Update mask
        if 'embeds' in track_queries and track_queries['embeds'].shape[0]>0 and mask.sum() > 0:
            """
            if not self.training:
                print('va a generar idxs para mantener y eliminar')
            """
            if self.training:
                ious = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['boxes'].squeeze(0)),box_cxcywh_to_xyxy(out_bboxes[-1][mask]))
                sim_idxs = torch.nonzero(ious > th_assoc)[:,1]
                sim_idxs_2 = torch.nonzero(torch.isin(torch.range(0,ious.shape[1]-1).to(device),sim_idxs,invert=True))
                sim_del = torch.nonzero(ious < th_assoc)[:,0]
                sim_keep = torch.nonzero(ious > th_assoc)[:,0]
            else:
                ious = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['boxes'].squeeze(0)),box_cxcywh_to_xyxy(out_bboxes[-1][mask]))
                print('ious:',ious)
                res = F.normalize(track_queries['cons_track']) @ F.normalize(track_queries['cons_det']).t()
                #print('cons_track:',track_queries['cons_track'])
                #print('cons_det:',track_queries['cons_det'])
                print('res:',res)
                sim_idxs = torch.nonzero(res > th_assoc)[:,1]
                sim_idxs_2 = torch.nonzero(torch.isin(torch.range(0,res.shape[1]-1).to(device),sim_idxs,invert=True))
                sim_del = torch.nonzero(res < th_assoc)[:,0]
                sim_keep = torch.nonzero(res > th_assoc)[:,0]
            #print('sim_idxs_2:',sim_idxs_2)

        if self.use_qim and mask.sum() and 'embeds' in track_queries and track_queries['embeds'].shape[0] > 0:
            """
            if not self.training:
                print('Usara modulo qim:')
                print('revisar:',output.shape)
                print('mask shape:',mask.shape)
                print('mask sum:',mask.sum())
                print('revisar:',output[mask.clone()].shape)
            """
            #print('entra por aqui:',track_queries['embeds'].shape)
            shape_split = [track_queries['embeds'].shape[1],output[mask].shape[0]]
            aux_queries = torch.cat((track_queries['embeds'],output[mask].unsqueeze(0)),dim=1)
            aux_boxes = torch.cat((track_queries['boxes'],out_bboxes[-1][mask].unsqueeze(0)),dim=1)
            #print('tracks:',track_queries['boxes'])
            #print('preds:',out_bboxes[-1][mask.clone()])
            if self.training:
                ious_mask = generalized_box_iou(box_cxcywh_to_xyxy(aux_boxes.squeeze(0)),box_cxcywh_to_xyxy(aux_boxes.squeeze(0)))
                #print('ious_mask:',ious_mask)
            else:
                
                cost_bbox = generalized_box_iou(box_cxcywh_to_xyxy(aux_boxes.squeeze(0)),box_cxcywh_to_xyxy(aux_boxes.squeeze(0)))
                combine_cons = torch.cat((track_queries['cons_track'],track_queries['cons_det']),dim=0)
                ious_mask = F.normalize(combine_cons) @ F.normalize(combine_cons).t()
                
                #cost_bbox = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['boxes'].squeeze(0)),box_cxcywh_to_xyxy(out_bboxes[-1][mask.clone()]))
                #combine_cons = torch.cat((track_queries['cons_track'],track_queries['cons_det']),dim=0)
                #ious_mask = F.normalize(combine_cons) @ F.normalize(combine_cons).t()
                #ious_mask = F.normalize(track_queries['cons_track']) @ F.normalize(track_queries['cons_det']).t()
                #print('cost_bbox:',cost_bbox)
                #print('res:',ious_mask)

            ious_mask = ious_mask < 0.5
            #print('dets:',out_bboxes[-1][mask.clone()])
            """
            ious_mask = ious_mask.clone().detach()
            track_queries['embeds_q'] = self.qim(aux_queries,self.query_pos_head(aux_boxes),ious_mask.clone())
            track_queries['embeds'], _ = torch.split(track_queries['embeds_q'], shape_split, dim=1)
            """
            #ious_mask = ious_mask[torch.sum(ious_mask,1)>0]
            ious_mask = ious_mask.detach()
            track_queries['embeds_q'] = self.qim(aux_queries,self.query_pos_head(aux_boxes),ious_mask)
            #track_queries['embeds_q'] = self.qim(self.query_pos_head(aux_boxes),ious_mask.clone())
            #track_queries['embeds'], _ = torch.split(track_queries['embeds_q'], shape_split, dim=1)
            track_queries['embeds_q'], _ = torch.split(track_queries['embeds_q'], shape_split, dim=1)
            aux_re = torch.sum(~ious_mask[:shape_split[0]],1)>1
            aux_emb = track_queries['embeds'].clone()
            aux_emb[:,aux_re.clone()] = track_queries['embeds_q'][:,aux_re.clone()]
            track_queries['embeds'] = aux_emb
            """
            if not self.training:
                print('boxes antes:',track_queries['boxes'])
            """
            aux_box = track_queries['boxes'].clone()
            aux_box[:,aux_re.clone()] = F.sigmoid(self.dec_bbox_head_track(track_queries['embeds'][:,aux_re.clone()])+inverse_sigmoid(track_queries['boxes'][:,aux_re.clone()].detach().clone())) #tener cuidado con este detach
            track_queries['boxes'] = aux_box
            """
            if not self.training:
                print('boxes despues:',track_queries['boxes'])
                print('scores antes de qim:',torch.sigmoid(track_queries['logits']))
            """
            aux_lo = track_queries['logits'].clone()
            aux_lo[:,aux_re.clone()] = self.dec_score_head_track(track_queries['embeds'][:,aux_re.clone()])
            track_queries['logits'] = aux_lo
            """
            if not self.training:
                print('scores despues de qim:',torch.sigmoid(track_queries['logits']))
            """
        if mask.sum() > 0:
            """
            if not self.training:
                print('actualiza la mascara con sim_idxs')
                print('sim_idxs:',sim_idxs)
            """
            mask[:,torch.unique(torch.nonzero(mask[0])[sim_idxs.to('cpu')])] = False
            
            #reid bloque para actualizar los consdet y mantener como consdet solo aquellas predicciones que no se detectan por un track
            if self.use_reid and sim_idxs_2.shape[0]>0 and track_queries['cons_det'] is not None:
                track_queries['cons_det'] = track_queries['cons_det'][torch.unique(sim_idxs_2)]
                track_queries['mask_pred'] = mask.clone() #update mask
            

        if criterion is not None:
            losses_dict, track_queries = criterion.match(out, track_queries, targets)

        #Delete
        if 'embeds' in track_queries and track_queries['embeds'].shape[0] > 0:
            #if not self.training:
                #print('aumentara el delete')
            if not self.training:
                track_mask  = torch.max(torch.sigmoid(track_queries['logits']),2)[0] > 0.3
                track_mask = track_mask.clone().detach()
                track_queries['delete'][track_mask.squeeze(0)] = 0
                track_queries['delete'][~track_mask.squeeze(0)] += 1
                #Utilizando la presencia de los detectores para la version 1
                if mask.sum() > 0:
                   track_queries['delete'][torch.unique(sim_del)] += 1
                   track_queries['delete'][torch.unique(sim_keep)] = 0
            else:
                if mask.sum() > 0:
                   track_queries['delete'][torch.unique(sim_del)] += 1
                   track_queries['delete'][torch.unique(sim_keep)] = 0

        #Add
        if mask.sum() > 0:
            """
            if not self.training:
               print('va a agregar')
               print('embeds antes:',track_queries['embeds'].shape)
            """
            track_queries['embeds'] = torch.cat((track_queries['embeds'],output[mask].unsqueeze(0)),dim=1)
            track_queries['logits'] = torch.cat((track_queries['logits'],out_logits[-1][mask].unsqueeze(0)),dim=1)
            track_queries['boxes'] = torch.cat((track_queries['boxes'],out_bboxes[-1][mask].unsqueeze(0)),dim=1)
            track_queries['delete'] = torch.cat((track_queries['delete'],torch.zeros(output[mask].shape[0]).to(device)),0)
            #if not self.training:
                #print('embeds despues:',track_queries['embeds'].shape)

            
            #Re-id Agregar los gtids de los tracks
            if self.use_reid and self.training:
                ious_hist = generalized_box_iou(box_cxcywh_to_xyxy(out_bboxes[-1][mask]),box_cxcywh_to_xyxy(targets[0]['boxes']))
                v, aux_idxs = torch.max(ious_hist,1)
                aux_idxs = aux_idxs[v>0.5]
                track_queries['gt_ids'] = torch.cat((track_queries['gt_ids'],targets[0]['ids'][aux_idxs]))
            
            
            #reid
            cont_ids_reid = 0
            if self.use_reid and 'ids_hist' in track_queries and track_queries['ids_hist'].shape[0] > 0 and mask.sum() > 0 and track_queries['cons_det'] is not None:
                aux_idxs_mask = torch.nonzero(mask[0])
                #aux_idxs_mask = aux_idxs_mask.to('cpu')
                if self.training:
                    ious_hist = generalized_box_iou(box_cxcywh_to_xyxy(out_bboxes[-1][mask]),box_cxcywh_to_xyxy(targets[0]['boxes']))
                    v, gt_idxs = torch.max(ious_hist,1)
                    v = v.to('cpu')
                    gt_idxs = gt_idxs[v>0.5]
                    aux_gt = targets[0]['ids'][gt_idxs]
                    mask_new_ids = torch.isin(track_queries['gt_ids_hist'],aux_gt)
                    mask_delete_ids = torch.isin(track_queries['gt_ids_hist'],aux_gt,invert=True)
                    print('ids hist:',track_queries['ids_hist'])
                    print('agrega del historico antes:',track_queries['ids'])
                    track_queries['ids'] = torch.cat((track_queries['ids'],track_queries['ids_hist'][mask_new_ids]))
                    cont_ids_reid=len(v>0.5)
                    print('cont_ids_reid:',cont_ids_reid)
                    print('agrega del historico despues:',track_queries['ids'])
                    print('revisar:',aux_idxs_mask[v>0.5])
                    print('mask shape:',mask.shape)
                    #mask = mask.to('cpu')
                    print('probandoo:',aux_idxs_mask[v>0.5])
                    if len(aux_idxs_mask[v>0.5])==0:
                        raise ValueError('aux_idxs_mask vacio')
                    print('revisar:',aux_idxs_mask[v>0.5])
                    #aux_mask = mask.clone()
                    #aux_mask[:,aux_idxs_mask[v>0.5]]=False
                    #mask[:,aux_idxs_mask[v>0.5]] = False
                    #mask[:,torch.unique(torch.nonzero(mask[0].clone())[v>0.5])] = False
                    #if len(aux_idxs_mask[v>0.5])>0:
                    #    mask[:,aux_idxs_mask[v>0.5]] = False
                    #Tal vez actualizar en esta parte la mask pred
                    print('mask_delete_ids:',mask_delete_ids)
                    
                    if mask_delete_ids.sum()>0:
                        """
                        mask_delete_ids = mask_delete_ids.to('cpu')
                        track_queries['ids_hist'] = track_queries['ids_hist'].to('cpu')
                        track_queries['projs_reid'] = track_queries['projs_reid'].to('cpu')
                        track_queries['gt_ids_hist'] = track_queries['gt_ids_hist'].to('cpu')
                        """
                        track_queries['ids_hist'] = track_queries['ids_hist'][mask_delete_ids]
                        track_queries['projs_reid'] = track_queries['projs_reid'][mask_delete_ids]
                        track_queries['gt_ids_hist'] = track_queries['gt_ids_hist'][mask_delete_ids]
                    else:
                        del track_queries['ids_hist']
                        del track_queries['projs_reid']
                        del track_queries['gt_ids_hist']
                    
                else:
                    th_reid = 0.5
                    aux_idxs = []
                    aux_idxs_0 = []
                    aux_cons_det = track_queries['cons_det'].clone()
                    res = F.normalize(aux_cons_det) @ F.normalize(track_queries['projs_reid']).t()
                    for j,aux in enumerate(res):
                        #print('aux:',aux)
                        v, idx = torch.max(aux,0)
                        if v>th_reid and idx not in aux_idxs:
                            aux_idxs.append(idx)
                            aux_idxs_0.append(j)
                    aux_idxs_0 = torch.tensor(aux_idxs_0,device=device)
                    aux_idxs = torch.tensor(aux_idxs,device=device)
                    if len(aux_idxs)>0:
                        print('agrega del historico antes:',track_queries['ids'])
                        track_queries['ids'] = torch.cat((track_queries['ids'],track_queries['ids_hist'][aux_idxs]))
                        print('agrega del historico despues:',track_queries['ids'])
                        cont_ids_reid=len(aux_idxs)
                        print('cont_ids_reid:',cont_ids_reid)

                    if len(aux_idxs_0)>0:
                        mask[:,aux_idxs_mask[aux_idxs_0]] = False

                    if len(aux_idxs_0)>0:
                        if len(aux_idxs)>len(track_queries['ids_hist']):
                            print('aux_idxs:',aux_idxs)
                            print('ids_hist:',track_queries['ids_hist'])
                            raise ValueError('Error de indices')

                    if len(aux_idxs)>0:
                        n_aux_idxs = torch.isin(track_queries['ids_hist'][aux_idxs],track_queries['ids_hist'],invert=True)
                        if n_aux_idxs.sum()>0:
                            track_queries['ids_hist'] = track_queries['ids_hist'][n_aux_idxs]
                            track_queries['projs_reid'] = track_queries['projs_reid'][n_aux_idxs]
                        else:
                            del track_queries['ids_hist']
                            del track_queries['projs_reid']

            print('cont_ids_reid:',cont_ids_reid)
            print('mask sum:',mask.sum())
            if self.use_reid and cont_ids_reid > 0:
                cont_ids = mask.sum() - cont_ids_reid
            else:
                cont_ids = mask.sum()
            print('cont_ids:',cont_ids)
            if "id_max" in track_queries:
                id_max = track_queries['id_max']
                aux_ids = torch.arange(id_max+1,id_max+1+cont_ids).to(device)
                print('ids antes:',track_queries['ids'])
                track_queries['ids'] = torch.cat((track_queries['ids'],aux_ids))
                print('ids despues:',track_queries['ids'])
            else:
                print('ids antes:',track_queries['ids'])
                track_queries['ids'] = torch.arange(0,cont_ids).to(device)
                print('ids despues:',track_queries['ids'])

            if 'history_id_max' not in track_queries:
                track_queries['history_id_max'] = torch.unique(track_queries['ids'])
            else:
                track_queries['history_id_max'] = torch.unique(torch.cat((track_queries['history_id_max'],track_queries['ids'])))
            track_queries['id_max'] = torch.max(track_queries['history_id_max'])


        if 'embeds' in track_queries and track_queries['embeds'].shape[0] > 0:
            #Update delete by scores
            #if not self.training:
                #print('scores:',torch.sigmoid(track_queries['logits']))
            aux_track_mask  = torch.max(torch.sigmoid(track_queries['logits']),2)[0] > 0.5
            aux_track_mask = aux_track_mask.squeeze(0)
            track_queries['delete'][(aux_track_mask == 0).nonzero().flatten()] += 1
            track_queries['delete'][aux_track_mask] = 0

            idxs_keep = torch.nonzero(track_queries['delete'] < limit).flatten()
            idxs_delete = torch.nonzero(track_queries['delete'] >= limit).flatten()
            print('delete:',track_queries['delete'])

        if 'embeds' in track_queries and track_queries['embeds'].shape[0] > 0 and idxs_delete.sum()>0:
            
            if self.use_reid:
                #save ids history
                if 'ids_hist' not in track_queries:
                    print('agregara ids hist:',track_queries['ids'][idxs_delete].clone())
                    track_queries['ids_hist'] = track_queries['ids'][idxs_delete].clone()

                else:
                    print('agregara ids hist:',track_queries['ids'][idxs_delete].clone())
                    track_queries['ids_hist'] = torch.cat((track_queries['ids_hist'],track_queries['ids'][idxs_delete].clone()))

                #projection values
                if 'projs_reid' not in track_queries:
                    track_queries['projs_reid'] = track_queries['cons_track'][idxs_delete].clone()
                else:
                    track_queries['projs_reid'] = torch.cat((track_queries['projs_reid'],track_queries['cons_track'][idxs_delete].clone()))

                if self.training:
                    #print('idxs_delete:',idxs_delete)
                    #print('gt_ids:',track_queries['gt_ids'])
                    if 'gt_ids_hist' not in track_queries:
                        track_queries['gt_ids_hist'] = track_queries['gt_ids'][idxs_delete]
                    else:
                        track_queries['gt_ids_hist'] = torch.cat((track_queries['gt_ids_hist'],track_queries['gt_ids'][idxs_delete]))
                
            
            track_queries['embeds'] = track_queries['embeds'][:,idxs_keep]
            track_queries['logits'] = track_queries['logits'][:,idxs_keep]
            track_queries['boxes'] = track_queries['boxes'][:,idxs_keep]
            print('delete antes:',track_queries['delete'])
            track_queries['delete'] = track_queries['delete'][idxs_keep]
            print('delete despues:',track_queries['delete'])

            track_queries['ids'] = track_queries['ids'][idxs_keep]

            #reid
            if self.use_reid and self.training:
                track_queries['gt_ids'] = track_queries['gt_ids'][idxs_keep] #reid
        """
        if not self.training:
            print('mask sum:',mask.sum())
            if 'embeds' in track_queries and track_queries['embeds'].shape[0]>0:
                print('como van las queries')
                print('embeds shape:',track_queries['embeds'].shape)
                print('boxes:',track_queries['boxes'].shape)
                print('scores:',torch.sigmoid(track_queries['logits']))
                print('ids:',track_queries['ids'].shape)
                print('delete:',track_queries['delete'])
        """
        
        #revisando
        if 'embeds' in track_queries and track_queries['embeds'].shape[0]>0:
            """
            print('como van las queries')
            print('embeds shape:',track_queries['embeds'].shape)
            print('boxes:',track_queries['boxes'].shape)
            print('scores:',torch.sigmoid(track_queries['logits']))
            print('ids:',track_queries['ids'])
            print('ids:',track_queries['ids'].shape)
            print('delete:',track_queries['delete'].shape)
            """
            if len(track_queries['ids']) != track_queries['embeds'].shape[1]:
                print('ids:',track_queries['ids'].shape)
                print('embeds:',track_queries['embeds'].shape)
                raise ValueError('inconsistencia ids y embeds')

            if 'ids_hist' in track_queries:
                if len(torch.unique(track_queries['ids_hist']))!=len(track_queries['ids_hist']):
                    print('ids_hist:',track_queries['ids_hist'])
                    raise ValueError('Error ids hist')
            
            if self.use_reid and self.training:
                """
                print('gt_ids:',track_queries['gt_ids'].shape)
                print('ids_hist:',track_queries['ids_hist'].shape)
                print('projs_reid shape:',track_queries['projs_reid'].shape)
                """
                if 'projs_reid' in track_queries and 'ids_hist' in track_queries and track_queries['projs_reid'].shape[0]!=track_queries['ids_hist'].shape[0]:
                    raise ValueError('Error de inconsistencia en los historicos')

        track_queries['cons_det'] = None
        if self.training:
            return losses_dict, track_queries
        else:
            return out, track_queries
