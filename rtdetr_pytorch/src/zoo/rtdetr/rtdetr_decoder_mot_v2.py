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


#torch.set_printoptions(threshold=10_000)

__all__ = ['RTDETRTransformerMOT_v2']

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
            #print("value shape:",value.shape)
            #print("value_mask shape:",value_mask.shape)
            value *= value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        #print('num_levels:',self.num_levels)
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

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        return output

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
        #print("query_pos_embed shape:",query_pos_embed.shape)
        #print("tgt:",tgt.shape)
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
           tgt2 = self.cross_attn(\
                   self.with_pos_embed(tgt, query_pos_embed), 
                   reference_points, 
                   memory, 
                   memory_spatial_shapes, 
                   memory_mask)
        else:
           #print("q shape:",q.shape)
           #print("k shape:",k.shape)
           #print("memory shape.",memory.shape)
           tgt2, attn_weights, attn_weights_logits = self.cross_attn(\
                   q,
                   memory,
                   memory)
           #print('attn_weights:',attn_weights)
        #print("tgt2 shape:",tgt2.shape)
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
                track_queries=None):

        output = tgt

        if track_queries['embeds'] is not None:
           det_tgt = output.clone()
           output = torch.cat((output,track_queries['embeds']),1)
           ref_points_unact = torch.cat((ref_points_unact,track_queries['track_ref_points']),1)
           if attn_mask is not None:
              aux_mask = torch.full([track_queries['embeds'].shape[1],det_tgt.shape[1]],False, dtype=torch.bool, device=attn_mask.device)
              aux_mask[:,:num_denoising] = True
              attn_mask = torch.cat((attn_mask,aux_mask),0)
              attn_mask = torch.cat((attn_mask,torch.full([attn_mask.shape[0],track_queries['embeds'].shape[1]],False, dtype=torch.bool, device=attn_mask.device)),1)
              #print('new attn_mask shape:',attn_mask.shape)
        
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        outputs = []
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)
            #pos_encoding.append(query_pos_embed)
            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed) #add memory_mask

            outputs.append(output)

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
            #ref_points_list.append(ref_points_detach)
            #ref_points_list.append(inverse_sigmoid(inter_ref_bbox)) #revisaar

        #outputs = torch.stack(outputs)
        #outputs = torch.tensor(outputs[-1])
        outputs = outputs[-1].clone().detach().requires_grad_(True)
        #ref_points_list = torch.stack(ref_points_list)
        #pos_encoding = torch.stack(pos_encoding)

        dec_out_bboxes = torch.stack(dec_out_bboxes)
        dec_out_logits = torch.stack(dec_out_logits)

        #dec_out_bboxes = torch.tensor(dec_out_bboxes[-1])
        #dec_out_logits = torch.tensor(dec_out_logits[-1])
        if track_queries['embeds'] is not None:
           outputs, track_outputs = torch.split(outputs,[det_tgt.shape[1],track_queries['embeds'].shape[1]],dim=1)
           dec_out_bboxes, track_dec_out_bboxes = torch.split(dec_out_bboxes, [det_tgt.shape[1],track_queries['embeds'].shape[1]],dim=2)
           dec_out_logits, track_dec_out_logits = torch.split(dec_out_logits, [det_tgt.shape[1],track_queries['embeds'].shape[1]],dim=2)
           track_queries['embeds_b'] = track_queries['embeds'].clone()
           #track_queries['embeds'] = track_outputs[-1]
           track_queries['embeds'] = track_outputs

           track_queries['track_dec_out_bboxes'] = track_dec_out_bboxes[-1]
           track_queries['track_dec_out_logits'] = track_dec_out_logits[-1]
           
           track_queries['track_dec_out_bboxes_b'] = track_dec_out_bboxes[-1].clone()
           track_queries['track_dec_out_logits_b'] = track_dec_out_logits[-1].clone()
           #track_queries['track_dec_out_bboxes'] = track_dec_out_bboxes
           #track_queries['track_dec_out_logis'] = track_dec_out_logits
        #track_queries['hs'] = outputs
        #track_queries["hs"] = outputs[-1]
        track_queries['hs'] = outputs
        return dec_out_bboxes, dec_out_logits, track_queries
"""
class TransformerDecoderMotTrack(nn.Module):
      def __init__(self, hidden_dim, decoder_layer, num_layers):
          super(TransformerDecoderMotTrack, self).__init__()
          self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
      def forward(self,tgt, memory, query_pos_embed, bbox_head=None, score_head=None, keep_head=None):
          dec_out_bboxes = []
          for i, layer in enumerate(self.layers):
              tgt = layer(tgt,None,memory,None,None,query_pos_embed=query_pos_embed)
              dec_out_bboxes.append(bbox_head(tgt))
              dec_out_logits.append(score_head(tgt))
              if i == len(self.layers)-1:
                 dec_out_keep = keep_head(tgt)
          return tgt, torch.stack(dec_out_logits), torch.stack(dec_out_bboxes), dec_out_keep
"""

class TransformerDecoderMotTrack(nn.Module):
      def __init__(self, hidden_dim, decoder_layer, num_layers):
          super(TransformerDecoderMotTrack, self).__init__()
          self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
          #self.self_attn = nn.MultiheadAttention(hidden_dim, 8, dropout)
      def forward(self,tgt, memory, query_pos_head_cons, boxes):
          #print('forma boxes:',boxes.shape)
          query_pos_embed = query_pos_head_cons(boxes)
          for i, layer in enumerate(self.layers):
              #print("lo que se envia tgt:",tgt.shape)
              #print("lo que se envia:",memory.shape)
              #print("lo que se envia query_pos_embed:",query_pos_embed.shape)
              tgt = layer(tgt,None,memory,None,None,query_pos_embed=query_pos_embed)
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

        #track
        track_decoder_layer = TransformerDecoderLayerMOT_v2(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points,deformable=False)
        self.track_decoder = TransformerDecoderMotTrack(hidden_dim, track_decoder_layer, 1)

        self.track_dec_keep_head = MLP(hidden_dim, hidden_dim, 2, num_layers=3)
        self.track_dec_add_head = MLP(hidden_dim, hidden_dim, 2, num_layers=3)

        #proj constr #probar
        self.enc_queries = nn.Sequential(
            nn.Linear(hidden_dim, num_queries),
            nn.LayerNorm(num_queries,)
        )

        self.query_pos_head_cons = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=num_decoder_layers)
        self.det_id_head_cons = MLP(num_queries, 2 * num_queries, num_queries, num_layers=6)

        #qim
        self.qim_int = QueryInteractionModule(hidden_dim,hidden_dim,hidden_dim)

        #hierarchie
        self.track_decoder_layer_h = TransformerDecoderLayerMOT_v2(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points,deformable=False)
        self.det_id_h = MLP(hidden_dim, 2 * hidden_dim, num_queries, num_layers=6)

    def forward(self, feats, targets=None, track_queries=None, ind_track=True, mode=''):
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
        out_bboxes, out_logits, track_queries = self.decoder(
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
            track_queries=track_queries)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            _ , track_queries['hs'] = torch.split(track_queries['hs'], dn_meta['dn_num_split'], dim=1)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        #out = {'pred_logits': out_logits, 'pred_boxes': out_bboxes}
        if mode != "cons" and "embeds" in track_queries and track_queries["embeds"] is not None:
           aux_track_queries_tgt, attn_weights, attn_weights_logits = self.track_decoder_layer_h(track_queries["embeds"],None,track_queries['hs'],None,None,None,None,self.query_pos_head(inverse_sigmoid(track_queries["track_dec_out_bboxes"])), return_attn=True)
           attn_weights = attn_weights.mean(0)
           attn_weights_logits = attn_weights_logits.mean(0)
           track_queries["embeds"] = aux_track_queries_tgt #probando uniendo los trackqueries
           #print("attn_weights shape:",attn_weights.shape)
           #print("attn_weights_logits shape:",attn_weights_logits.shape)
           track_queries['attn_weights'] = attn_weights_logits
           #print("aux queries shape:",aux_track_queries_tgt.shape)
           track_queries["det_idxs"] = self.det_id_h(aux_track_queries_tgt.squeeze(0))
        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta
        """
        if self.training:
           th = 0.5
        else:
           th = 0.20
        """
        th = 0.7
        #th = 0.5
        #mask = torch.max(out_logits[-1],2)[0] > th
        with torch.no_grad():
             #scores = F.softmax(out_logits[-1],dim=2)
             scores = F.sigmoid(out_logits[-1])
        #print('scores:',scores)
        mask = torch.max(scores,2)[0] > th
        #print("llegaa")
        if not self.training:
           #print('scores:',scores)
           if mask.sum()>0:
              print('objetos detectados:',mask.sum())
        """ proceso de cons
        if mode == 'cons' and ('embeds_cons' not in track_queries or track_queries['embeds_cons'] is None):
           if mask.sum() > 0:
              #track_queries['embeds_cons'] = track_queries['hs'][-1][mask]
              track_queries['embeds_cons'] = track_queries['hs'][mask]
              track_queries['track_proj'] = self.enc_queries(track_queries['embeds_cons'])
              track_queries['track_boxes'] = out['pred_boxes'].clone()[mask]
              #track_queries['det_queries'] = track_queries['hs'][-1][mask].clone()
              #print("agrega queries:",track_queries["embeds_cons"].shape[0])
              #print("entra primera")

        if mode == 'cons' and 'embeds_cons' in track_queries and track_queries['embeds_cons'] is not None:
           #print("entra segunda")
           if 'rem_ids' in track_queries and track_queries['rem_ids'] is not None and len(track_queries['rem_ids'])>0:
              #print('rem mask:',track_queries['rem_ids'].shape)
              #print('embeds_cons antes de eliminar:',track_queries['embeds_cons'].shape)
              track_queries['embeds_cons'] = track_queries['embeds_cons'][track_queries['rem_ids']]
              #print('embeds_cons despues de eliminar:',track_queries['embeds_cons'].shape)
              if track_queries['embeds_cons'].shape[0]==0:
                 #print('Pone en None los tracks')
                 track_queries['embeds_cons'] = None
                 track_queries['track_proj'] = None
                 track_queries['track_boxes'] = None
                 track_queries['det_queries'] = None
                 track_queries['det_proj'] = None
                 track_queries['det_ids'] = None
                 track_queries['rem_ids'] = None
                 track_queries['add_ids'] = None
                 track_queries['replace_ids'] = None

           if 'replace_ids' in track_queries and track_queries['replace_ids'] is not None and len(track_queries['replace_ids'])>0:
               #print("replace_ids:",track_queries["replace_ids"])
               track_queries['embeds_cons'] = track_queries['det_queries'][track_queries["replace_ids"][:,1]]
               track_queries['track_proj'] = self.enc_queries(track_queries['embeds_cons'])
               track_queries['track_boxes'] = track_queries['det_boxes'][track_queries["replace_ids"][:,1]]
               
           if 'add_ids' in track_queries and track_queries['add_ids'] is not None and len(track_queries['add_ids'])>0:
              #print('embeds_cons before shape:',track_queries['embeds_cons'].shape)
              #print('add_ids:',track_queries['add_ids'])
              #print('det_queries shape:',track_queries['det_queries'].shape)
              aux_det_queries = torch.index_select(track_queries['det_queries'], 0, track_queries['add_ids'])
              #print('embeds_cons before shape:',track_queries['embeds_cons'].shape)
              track_queries['embeds_cons'] = torch.cat((track_queries['embeds_cons'],aux_det_queries))
              #print('embeds_cons after shape:',track_queries['embeds_cons'].shape)
              #print('track_boxes before shape:',track_queries['track_boxes'].shape)
              track_queries['track_boxes'] = torch.cat((track_queries['track_boxes'],track_queries['det_boxes'][track_queries['add_ids']].clone()))
              #print('track_boxes after shape:',track_queries['track_boxes'].shape)
              track_queries['track_proj'] = self.enc_queries(track_queries['embeds_cons'])
              #print("adiciona")
           
           if mask.sum() > 0 and track_queries['embeds_cons'] is not None:
              #det_queries = track_queries['hs'][-1][mask]
              det_queries = track_queries['hs'][mask]
              #det_queries = track_queries["hs"][-1].squeeze(0)
              #print('det_queries shape:',det_queries.shape)
              track_queries['det_queries'] = det_queries
              #track_queries['det_boxes'] = out['pred_boxes'].clone()[mask]
              track_queries['det_boxes'] = out['pred_boxes'][mask].clone()
              track_queries['embeds_cons'] = self.track_decoder(track_queries['embeds_cons'],det_queries,self.query_pos_head_cons,track_queries['track_boxes'])
              #print("sale del decoder:",track_queries["embeds_cons"].shape)
              track_queries['track_proj'] = self.enc_queries(track_queries['embeds_cons'])
              track_queries['det_proj'] = self.enc_queries(det_queries)
              track_queries['det_ids'] = self.det_id_head_cons(track_queries['track_proj'])
              #print('det_ids:',torch.argmax(track_queries['det_ids'],1))
              #print('det_ids:',track_queries['det_ids'].shape)
              track_queries['rem_ids'] = torch.tensor([False if torch.argmax(ids) > det_queries.shape[0] else True for i,ids in enumerate(track_queries['det_ids'])])
              aux_det_ids = torch.argmax(track_queries['det_ids'],dim=1)
              track_queries['add_ids'] = torch.tensor([i for i in range(track_queries['det_queries'].shape[0]) if i not in aux_det_ids]).to(det_queries.device)
              track_queries['replace_ids'] = torch.tensor([[i,torch.argmax(ids)] for i,ids in enumerate(track_queries['det_ids']) if torch.argmax(ids) < det_queries.shape[0]])
           else:
              track_queries['embeds_cons'] = None
              track_queries['track_proj'] = None
              track_queries['track_boxes'] = None
              track_queries['det_queries'] = None
              track_queries['det_proj'] = None
              track_queries['det_ids'] = None
              track_queries['rem_ids'] = None
              track_queries['add_ids'] = None
              track_queries['replace_ids'] = None
              track_queries["id_max"] = None
           #if mask.sum() > 0:
           #track_queries['embeds_cons'] = torch.cat((track_queries['embeds_cons'],det_queries[mask_cons_new]))
           #split[keep_mask].unsqueeze(0)
        """
        if track_queries['embeds'] is None and ind_track and mode != 'cons':
           if mask.sum()>0:
              #det_embeds = track_queries['hs'][-1][mask]
              det_embeds = track_queries['hs'][mask]
              ref_points = inverse_sigmoid(out_bboxes[-1][mask].unsqueeze(0).detach().clone())
              track_queries['embeds'] = det_embeds.unsqueeze(0)
              track_queries['track_ref_points'] = ref_points
              track_queries['pred_logits'] = out_logits[-1][mask].unsqueeze(0) #.clone()
              track_queries['pred_boxes'] = out_bboxes[-1][mask].unsqueeze(0) #.clone()
              if "id_max" not in track_queries:
                 track_queries['ids'] = torch.arange(0,det_embeds.shape[0]).to(track_queries['embeds'].device)
              else:
                 id_max = track_queries['id_max']
                 track_queries['ids'] = torch.arange(id_max+1,id_max+1+det_embeds.shape[0]).to(track_queries["embeds"].device)
              track_queries['id_max'] = torch.max(track_queries["ids"])
              track_queries['delete'] = torch.zeros(det_embeds.shape[0]).to(track_queries['embeds'].device)
              if not self.training:
                 print('agrega queries para empezar:',det_embeds.shape)
                 #print('ref_points revisar:',track_queries['track_ref_points'])
              print('agrega queries para empezar:',det_embeds.shape)
        else:
          if 'attn_weights' in track_queries:
                  attn_th = 0.5
                  """
                  if self.training:
                     limit = 1
                  else:
                     limit = 5
                  """
                  limit = 7 #3
                  #max_attn, max_idxs_det = torch.topk(track_queries['attn_weights'],10)  como estaba antes
                  with torch.no_grad():
                       max_attn_top, max_idxs_det_top = torch.topk(torch.sigmoid(track_queries['det_idxs']), 10)
                       max_attn, max_idxs_det = torch.topk(torch.sigmoid(track_queries['attn_weights']),10) #group 10
                  if not self.training:
                     #print("max_attn_top:",max_attn_top)
                     #print("max_ids_det_top:",max_idxs_det_top)
                     print("max_ids_det_top_attn:",torch.topk(torch.sigmoid(track_queries["attn_weights"]),20)[1])
                  """
                  with torch.no_grad():
                       print('sin sigmoid:',track_queries["attn_weights"])
                       print('sigmoid todod:',torch.sigmoid(track_queries["attn_weights"]))
                       print("max_ids_det_top_attn:",torch.topk(torch.sigmoid(track_queries["attn_weights"]),10)[1]) #quitar
                  """
                  #max_attn_top, max_idxs_det_top = torch.topk(track_queries['attn_weights'], 10) #4
                  #max_attn, max_idxs_det = torch.max(track_queries['attn_weights'],dim=-1) #group n
                  #print("attn_weights:",track_queries['attn_weights'])
                  #print('max_attn:',max_attn)
                  #mask_attn = torch.max(max_attn_top,dim=-1)[0] > attn_th
                  mask_attn = torch.zeros(track_queries['delete'].shape)
                  #print('mask attn shape:',mask_attn.shape)
                  #print('delete shape:',track_queries['delete'].shape)
                  if mask.sum() > 0:
                     #print('max_idxs_det_top shape:',max_idxs_det_top.shape)
                     #print('nonzero shape:',torch.nonzero(mask).shape)
                     #print('mask:',mask)
                     #print('isin:',torch.sum(torch.isin(max_idxs_det_top,torch.nonzero(mask[0])),dim=1)>0)
                     #mask_attn = torch.sum(torch.isin(max_idxs_det_top,torch.nonzero(mask[0])),dim=1)>0
                     mask_attn = torch.sum(torch.isin(max_idxs_det,torch.nonzero(mask[0])),dim=1)>0
                  else:
                     mask_attn = mask_attn > 0
                  #if not self.training:
                  #   print("ids_det:",torch.nonzero(mask_attn))
                  #print('mask_atnn:',mask_attn)
                  #print('delete antes:',track_queries['delete'])
                  #track_queries['delete'][mask_attn.squeeze(0)]=0
                  #track_queries['delete'][~mask_attn.squeeze(0)]+=1
                  track_queries['delete'][mask_attn]=0
                  track_queries['delete'][~mask_attn]+=1
                  #print('delete despues:',track_queries['delete'])
                  mask_delete = track_queries['delete'] > limit
                  #print('delete limit:',track_queries['delete'])
                  #print('mask delete:',mask_delete)
                  mask_delete = ~mask_delete.unsqueeze(0)
                  track_queries['embeds'] = track_queries['embeds'][mask_delete].unsqueeze(0)
                  track_queries["embeds_b"] = track_queries['embeds_b'][mask_delete].unsqueeze(0)
                  track_queries['pred_logits'] = track_queries["track_dec_out_logits"][mask_delete].unsqueeze(0)
                  track_queries['pred_boxes'] = track_queries["track_dec_out_bboxes"][mask_delete].unsqueeze(0)
                  track_queries['track_ref_points_b'] = track_queries["pred_boxes"].detach().clone()
                  track_queries['track_ref_points'] = inverse_sigmoid(track_queries["pred_boxes"].detach().clone())
                  track_queries["ids"] = track_queries['ids'][mask_delete.squeeze(0)]
                  track_queries['delete'] = track_queries['delete'][mask_delete.squeeze(0)]
                  #print('forma embeds:',track_queries['embeds'].shape)
                  if track_queries['embeds'].shape[1]==0:
                     track_queries['embeds'] = None
                     track_queries["embeds_b"] = None
                     track_queries['pred_logits'] = None
                     track_queries['pred_boxes'] = None
                     track_queries['track_ref_points'] = None
                     track_queries['track_ref_points_b'] = None
                     track_queries["ids"] = None
                     track_queries["delete"] = None
                     if not self.training:
                        print("track queries se han puesto vacias")
                  else:
                     track_queries = self.qim_int(track_queries)
                  with torch.no_grad():
                       #mask_add = torch.max(F.softmax(out_logits[-1],dim=-1),dim=-1)[0] > th #-> con esta mascara agregar las nuevas consultas
                       #print("scores softmax:",torch.max(F.softmax(out_logits[-1],dim=-1),dim=-1)[0])
                       #print("scores sigmoid:",torch.max(F.sigmoid(out_logits[-1]),dim=-1)[0])
                       mask_add = torch.max(F.sigmoid(out_logits[-1]),dim=-1)[0] > th
                       #print('mask add a revisar:',mask_add)
                       #print('max_attn:',max_attn.shape)
                       #print('max_idxs_det shape:',max_idxs_det.shape)
                       #print("max_idxs_det:",max_idxs_det[mask_attn])
                       if mask_add.sum()>0:
                          if not self.training:
                             print("mask_add shape:",mask_add.shape)
                             print("ids_det:",torch.nonzero(mask_add[0]))
                          print("ids_det:",torch.nonzero(mask_add[0]))
                          track_queries['ids_det'] = torch.nonzero(mask_add[0])
                          #print("mask_add antes:",mask_add.sum())
                          #print("mask_add:",mask_add)
                          #print("indices de las detecciones que va a agregar:",mask_add[0].nonzero())
                          #print("indices detectados por attn:",max_idxs_det[mask_attn])
                          #print("indices top attn:",max_idxs_det_top[mask_attn])
                          #mask_add[:,max_idxs_det[mask_attn]]=False
                          #mask_add[:,torch.unique(max_idxs_det_top[mask_attn].flatten())]=False como estaba antes
                          mask_add[:,torch.unique(max_idxs_det.flatten())]=False
                          #print("mask_add despues:",mask_add.sum())
                       #print('actualiza y pasaa')
                       #print('despues:',mask_add)
                  if mask_add.sum() > 0:
                     if not self.training:
                        print('va a agregaar')
                        print('mask_add:',mask_add.sum())
                     #print('cantidad:',mask_add.sum())
                     #print('logits shape:',out_logits[-1].shape)
                     #print('softmax logits:',F.softmax(out_logits[-1],dim=-1))
                     #print('scores:',torch.max(F.softmax(out_logits[-1],dim=-1),dim=-1)[0])
                     #print("mask_add:",mask_add)
                     if track_queries["embeds"] is not None:
                        track_queries['embeds'] = torch.cat((track_queries['embeds'],track_queries['hs'][mask_add].unsqueeze(0)),dim=1)
                        track_queries['pred_logits'] = torch.cat((track_queries['pred_logits'],out_logits[-1][mask_add].unsqueeze(0)),dim=1)
                        track_queries['pred_boxes'] = torch.cat((track_queries['pred_boxes'],out_bboxes[-1][mask_add].unsqueeze(0)),dim=1)
                        track_queries['track_ref_points_b'] = track_queries["pred_boxes"].detach().clone()
                        track_queries['track_ref_points'] = inverse_sigmoid(track_queries["pred_boxes"].detach().clone())
                        id_max = track_queries['id_max']
                        track_queries["ids"] = torch.cat((track_queries["ids"],torch.arange(id_max+1,id_max+1+mask_add.sum()).to(track_queries["embeds"].device)))
                        track_queries["id_max"] = torch.max(track_queries["ids"])
                        track_queries["delete"] = torch.cat((track_queries["delete"],torch.zeros(mask_add.sum()).to(track_queries['embeds'].device)))
                     else:
                        track_queries['embeds'] = track_queries['hs'][mask_add].unsqueeze(0)
                        track_queries['pred_logits'] = out_logits[-1][mask_add].unsqueeze(0)
                        track_queries['pred_boxes'] = out_bboxes[-1][mask_add].unsqueeze(0)
                        track_queries['track_ref_points_b'] = track_queries["pred_boxes"].detach().clone()
                        track_queries['track_ref_points'] = inverse_sigmoid(track_queries["pred_boxes"].detach().clone())
                        id_max = track_queries["id_max"]
                        track_queries["ids"] = torch.arange(id_max+1,id_max+1+mask_add.sum()).to(track_queries["embeds"].device)
                        track_queries["id_max"] = torch.max(track_queries["ids"])
                        track_queries["delete"] = torch.zeros(mask_add.sum()).to(track_queries['embeds'].device)
                     #print("ids despues de agregar:",track_queries["ids"])
        if not self.training:
           if track_queries['embeds'] is not None:
              #print('embeds:',track_queries['embeds'])
              print('como van los track queries:',track_queries['embeds'].shape)
              #print('boxes:',track_queries['pred_boxes'])
              #print('ref_points:',track_queries['track_ref_points'])
        """
        else:
           if ind_track and mode != 'cons':
              track_queries["keep"] = self.track_dec_keep_head(track_queries["embeds"])
              with torch.no_grad():
                   idxs_mask = torch.argmax(F.sigmoid(track_queries['keep']),dim=2)
              #idxs_mask = torch.argmax(track_queries['keep'],dim=2)
              keep_mask = idxs_mask == 1
              #if not self.training:
                 #print('keep:',track_queries['keep'])
                 #print('keep:',keep_mask.sum())
                 #print("keep:",keep_mask)
              #print('keep:',track_queries['keep'])
              #print('keep:',keep_mask)
              track_queries['embeds'] = track_queries['embeds'][keep_mask].unsqueeze(0) #probar
              if track_queries['embeds'].shape[1]==0:
                 #print('Pone queries en None')
                 track_queries['embeds'] = None
                 track_queries['pred_logits'] = None
                 track_queries['pred_boxes'] = None
                 track_queries['track_ref_points'] = None
                 track_queries['track_ref_points_b'] = None
              #for return values
              else:
                 track_queries["pred_logits"] = track_queries["track_dec_out_logits"][keep_mask].unsqueeze(0)
                 track_queries["pred_boxes"] = track_queries["track_dec_out_bboxes"][keep_mask].unsqueeze(0)
                 track_queries['track_ref_points_b'] = track_queries['track_ref_points'][keep_mask].unsqueeze(0).clone()
                 track_queries['track_ref_points'] = inverse_sigmoid(track_queries["pred_boxes"].detach().clone())
                 track_queries['embeds_b'] = track_queries['embeds_b'][keep_mask].unsqueeze(0)
                 #print("keep mask shape:",keep_mask.shape)
                 #print("ids shape:",track_queries["ids"].shape)
                 #print("keep_mask device:",keep_mask.device)
                 #print("embeds depsues de keepmask shape:",track_queries["embeds"].shape)
                 #print("track_queries ids before:",track_queries["ids"])
                 #print("keep_mask:",keep_mask)
                 track_queries["ids"] = track_queries["ids"][keep_mask.squeeze(0)]
                 #print("track_queries ids after:",track_queries["ids"])
                 #use qim
                 #print('track_queries que envia a qim:',track_queries)
                 track_queries = self.qim_int(track_queries)

              if mask.sum()>0:
                 #track_queries["add"] = self.track_dec_add_head(track_queries["hs"][-1][mask].unsqueeze(0))
                 track_queries["add"] = self.track_dec_add_head(track_queries["hs"][mask].unsqueeze(0))
                 track_queries['add_mask'] = mask
                 #with torch.no_grad():
                 idxs_mask_dets = torch.argmax(F.sigmoid(track_queries['add']),dim=2)
                 #idxs_mask_dets = torch.argmax(track_queries['add'],dim=2)
                 add_mask = idxs_mask_dets == 1
                 add_mask = add_mask.squeeze(0)
                 #if not self.training:
                 #   print('add_mask:',add_mask)
                 #print('forma track queries:',track_queries['embeds'].shape)
                 #print('detetions to add:',track_queries['hs'][-1][mask][add_mask].shape)
                 if track_queries['embeds'] is not None and add_mask.sum()>0:
                    #print("track queries antes de adicionar:",track_queries['embeds'].shape)
                    #track_queries['embeds'] = torch.cat((track_queries['embeds'],track_queries['hs'][-1][mask][add_mask].unsqueeze(0)),dim=1)
                    track_queries['embeds'] = torch.cat((track_queries['embeds'],track_queries['hs'][mask][add_mask].unsqueeze(0)),dim=1)
                    track_queries['pred_logits'] = torch.cat((track_queries['pred_logits'],out_logits[-1][mask][add_mask].unsqueeze(0)),dim=1)
                    track_queries['pred_boxes'] = torch.cat((track_queries['pred_boxes'],out_bboxes[-1][mask][add_mask].unsqueeze(0)),dim=1)
                    track_queries['track_ref_points'] = inverse_sigmoid(track_queries["pred_boxes"].detach().clone())
                    id_max = track_queries["id_max"]
                    #print("como van los ids:",track_queries["ids"])
                    #print("rango:",id_max+1,id_max+1+add_mask.sum())
                    #print("ids:",track_queries["ids"])
                    track_queries["ids"] = torch.cat((track_queries["ids"],torch.arange(id_max+1,id_max+1+add_mask.sum()).to(track_queries["embeds"].device)))
                    track_queries["id_max"] = torch.max(track_queries["ids"])
                    #print("agrega objeto")
                    #print("como van los ids:",track_queries["ids"])
                    #print("adiciona 1")
                 else:
                    if add_mask.sum()>0:
                       #track_queries['embeds'] = track_queries['hs'][-1][mask][add_mask].unsqueeze(0)
                       track_queries['embeds'] = track_queries['hs'][mask][add_mask].unsqueeze(0)
                       #print('agrega new queries with empty track queries:',track_queries['embeds'].shape)
                       track_queries['pred_logits'] = out_logits[-1][mask][add_mask].unsqueeze(0)
                       track_queries['pred_boxes'] = out_bboxes[-1][mask][add_mask].unsqueeze(0)
                       track_queries['track_ref_points'] = inverse_sigmoid(track_queries["pred_boxes"].detach().clone())
                       track_queries["ids"] = torch.arange(add_mask.sum()).to(track_queries["embeds"].device)
                       track_queries["id_max"] = torch.max(track_queries["ids"])
                       #print("agrega objeto")
                       #print("como van los ids:",track_queries["ids"])
                       #print("adiciona 2")
              if not self.training:
                 if track_queries['embeds'] is not None:
                    #print('embeds:',track_queries['embeds'])
                    print('como van los track queries:',track_queries['embeds'].shape)
                    #print('boxes:',track_queries['pred_boxes'])
                    #print('ref_points:',track_queries['track_ref_points'])
              #if track_queries['embeds'] is not None:
              #   print('como van los track queries:',track_queries['embeds'].shape)
              #   print("como van los ids:",track_queries["ids"])
              #   print('boxes:',track_queries['pred_boxes'])
              #print('ref_points:',track_queries['track_ref_points'])
        """
        return out, track_queries
