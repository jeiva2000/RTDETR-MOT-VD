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


__all__ = ['RTDETRTransformerMOT_v2']


def max_indices_by_group(ids, ious):
    return {unique_id.item(): (ious[ids == unique_id].argmax() + (ids == unique_id).nonzero(as_tuple=True)[0][0]).item() 
            for unique_id in torch.unique(ids)}

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


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        
        self.final_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # Scale dot product
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        if key_padding_mask is not None:
            # Assuming key_padding_mask is a ByteTensor with shape [batch_size, seq_len] where padding elements are True
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        #attn = F.softmax(scores, dim=-1)
        attn = F.sigmoid(scores)
        attn = self.dropout(attn)
        output = torch.matmul(attn, value)
        return output

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size = query.size(0)
        
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = self.scaled_dot_product_attention(query, key, value, attn_mask, key_padding_mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.final_linear(attn_output)
        
        return output

class QIM(nn.Module):
    def __init__(self,hidden_dim=256,dropout=0.2):
        super(QIM, self).__init__()
        #self.self_attn = nn.MultiheadAttention(hidden_dim, 8, dropout,batch_first=True) #anterior
        self.self_attn = MultiheadAttention(hidden_dim, 8)
        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU(True)
    """
    def forward(self,qdt,boxes,mask): #revisaar
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
    """
    def forward(self,qdt,boxes,mask): #revisaar
        query_pos = boxes
        q = k = tgt = query_pos + qdt
        #tgt2, _ = self.self_attn(q,k,tgt,attn_mask=mask)
        tgt2 = self.self_attn(q,k,tgt,attn_mask=mask)
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
        self.mode = 0
        self.use_qim = False #True #Pendientee de esta partee
        self.use_reid = False #True #False #True
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

        if self.use_qim:
            self.qim = QIM()
            self.dec_score_head_track = nn.Linear(hidden_dim, num_classes) 
            self.dec_bbox_head_track = MLP(hidden_dim, hidden_dim, 4, 3)

        """
        #Time Embed
        self.embeds_t = nn.Embedding(1000, hidden_dim) #Probar rapidooo
        self.use_embeds_t = True
        """

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

        if 'embeds' in track_queries and track_queries['embeds'] is not None and ind_track:
            det_num_split = out_bboxes.shape[2]-t_num_split
            
            output, track_queries['embeds'] = torch.split(output,[det_num_split,t_num_split],dim=1)
            out_bboxes, track_queries['boxes'] = torch.split(out_bboxes,[det_num_split,t_num_split],dim=2)
            track_queries['boxes'] = track_queries['boxes'][-1]
            out_logits, track_queries['logits'] = torch.split(out_logits,[det_num_split,t_num_split],dim=2)
            track_queries['logits'] = track_queries['logits'][-1]
            
            if self.mode in [2,3]:
                sampling_values_det, sampling_values_track = torch.split(sampling_values_det, [det_num_split,t_num_split], dim=3)

        out = {'embeds':self.proj_cons(output), 'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        
        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        if self.training:
            outputs_without_aux = {k: v for k, v in out.items() if 'aux' not in k}
            indices = criterion.matcher(outputs_without_aux, targets)

        if 'ids_gt' not in track_queries and self.training:
            if len(indices[0][1]) > 0:
                track_queries['ids_gt'] = targets[0]['ids'][indices[0][1]]
                track_queries['embeds'] = output[:,indices[0][0]]
                track_queries['boxes'] = out_bboxes[-1][:,indices[0][0]]
                track_queries['logits'] = out_logits[-1][:,indices[0][0]]
                track_queries['ids'] = torch.arange(0,output[:,indices[0][0]].shape[1]).to(device)
                track_queries['id_max'] = torch.max(track_queries['ids'])
                track_queries['delete'] = torch.zeros(track_queries['embeds'].shape[1]).to(device)

        elif self.training and 'delete' in track_queries:
            aux_del = torch.isin(track_queries['ids_gt'],targets[0]['ids'],invert=True)
            track_queries['delete'][~aux_del]=0
            track_queries['delete'][aux_del]+=1
            #print('dist:',F.normalize(out['embeds'].squeeze(0)) @ F.normalize(self.proj_cons(track_queries['embeds']).squeeze(0)).t())
            aux_idxs = torch.nonzero(torch.isin(targets[0]['ids'][indices[0][1]],track_queries['ids_gt'],invert=True)).flatten()
            if len(aux_idxs) > 0:
                track_queries['ids_gt'] = torch.cat((track_queries['ids_gt'],targets[0]['ids'][aux_idxs]))
                for i, idx in enumerate(indices[0][1]):
                    if idx in aux_idxs:
                        track_queries['embeds'] = torch.cat((track_queries['embeds'],output[:,indices[0][0][i]].unsqueeze(0)),dim=1)
                        track_queries['boxes'] = torch.cat((track_queries['boxes'],out_bboxes[-1][:,indices[0][0][i]].unsqueeze(0)),dim=1)
                        track_queries['logits'] = torch.cat((track_queries['logits'],out_logits[-1][:,indices[0][0][i]].unsqueeze(0)),dim=1)
                track_queries['ids'] = torch.cat((track_queries['ids'],torch.arange(track_queries['id_max']+1,track_queries['id_max']+1+len(aux_idxs)).to(device)))
                track_queries['id_max'] = torch.max(track_queries['ids'])
                track_queries['delete'] = torch.cat((track_queries['delete'],torch.zeros(len(aux_idxs)).to(device)))

        if not self.training:
            aux_mask = torch.max(torch.sigmoid(out_logits[-1]),2)[0][0]>0.5
            aux_mask = aux_mask.to(device)
            if 'embeds' not in track_queries or track_queries['embeds'].shape[1]==0:
                if aux_mask.sum() > 0:
                    track_queries['embeds'] = output[:,aux_mask]
                    track_queries['boxes'] = out_bboxes[-1][:,aux_mask]
                    track_queries['logits'] = out_logits[-1][:,aux_mask]
                    track_queries['delete'] = torch.zeros(track_queries['embeds'].shape[1]).to(device)
                    track_queries['ids'] = torch.arange(0,out_bboxes[-1][:,aux_mask].shape[1]).to(device)
                    track_queries['id_max'] = torch.max(track_queries['ids'])
            elif track_queries['embeds'].shape[1]>0:
                dist = F.normalize(out['embeds'].squeeze(0)) @ F.normalize(self.proj_cons(track_queries['embeds']).squeeze(0)).t()
                aux_idxs = torch.max(dist,1)[0]<0.5
                aux_idxs = aux_idxs.flatten()
                aux_idxs = aux_idxs&aux_mask
                aux_mask_del = torch.max(torch.sigmoid(track_queries['logits']),2)[0][0]>0.5
                track_queries['delete'][aux_mask_del]=0
                track_queries['delete'][~aux_mask_del]+=1
                if aux_idxs.sum() > 0:
                    track_queries['embeds'] = torch.cat((track_queries['embeds'],output[:,aux_idxs]),dim=1)
                    track_queries['boxes'] = torch.cat((track_queries['boxes'],out_bboxes[-1][:,aux_idxs]),dim=1)
                    track_queries['logits'] = torch.cat((track_queries['logits'],out_logits[-1][:,aux_idxs]),dim=1)
                    track_queries['delete'] = torch.cat((track_queries['delete'],torch.zeros(aux_idxs.sum()).to(device)))
                    track_queries['ids'] = torch.cat((track_queries['ids'],torch.arange(track_queries['id_max']+1,track_queries['id_max']+1+aux_idxs.sum()).to(device)))
                    track_queries['id_max'] = torch.max(track_queries['ids'])

        if 'delete'in track_queries and track_queries['delete'].shape[0] > 0:
            aux_mask_del = track_queries['delete']<3
            track_queries['embeds'] = track_queries['embeds'][:,aux_mask_del]
            track_queries['boxes'] = track_queries['boxes'][:,aux_mask_del]
            track_queries['logits'] = track_queries['logits'][:,aux_mask_del]
            track_queries['delete'] = track_queries['delete'][aux_mask_del]
            track_queries['ids'] = track_queries['ids'][aux_mask_del]
            if self.training:
                track_queries['ids_gt'] = track_queries['ids_gt'][aux_mask_del]

        if 'embeds' in track_queries and track_queries['embeds'].shape[1] > 0:
            track_queries['cons_embs'] = self.proj_cons(track_queries['embeds'])
        
        if self.training:
            losses_dict, track_queries = criterion.match(out, targets, track_queries)
            return losses_dict, track_queries
        else:
            return out, track_queries
