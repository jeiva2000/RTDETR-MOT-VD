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
                track_queries=None,
                mode=0):

        output = tgt

        """
        if track_queries['embeds'] is not None and mode == 0:
           det_tgt = output.clone()
           output = torch.cat((output,track_queries['embeds']),1)
           ref_points_unact = torch.cat((ref_points_unact,track_queries['track_ref_points']),1)
           if attn_mask is not None:
              aux_mask = torch.full([track_queries['embeds'].shape[1],det_tgt.shape[1]],False, dtype=torch.bool, device=attn_mask.device)
              aux_mask[:,:num_denoising] = True
              attn_mask = torch.cat((attn_mask,aux_mask),0)
              attn_mask = torch.cat((attn_mask,torch.full([attn_mask.shape[0],track_queries['embeds'].shape[1]],False, dtype=torch.bool, device=attn_mask.device)),1)
              #print('new attn_mask shape:',attn_mask.shape)
        """#Comentado ya que no se usa
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

        #outputs = outputs[-1].clone()#.detach().requires_grad_(True)
        outputs = outputs[-1]

        dec_out_bboxes = torch.stack(dec_out_bboxes)
        dec_out_logits = torch.stack(dec_out_logits)

        """
        if track_queries['embeds'] is not None and mode == 0:
           outputs, track_outputs = torch.split(outputs,[det_tgt.shape[1],track_queries['embeds'].shape[1]],dim=1)
           dec_out_bboxes, track_dec_out_bboxes = torch.split(dec_out_bboxes, [det_tgt.shape[1],track_queries['embeds'].shape[1]],dim=2)
           dec_out_logits, track_dec_out_logits = torch.split(dec_out_logits, [det_tgt.shape[1],track_queries['embeds'].shape[1]],dim=2)
           track_queries['embeds_b'] = track_queries['embeds'].clone()
           track_queries['embeds'] = track_outputs

           track_queries['track_dec_out_bboxes'] = track_dec_out_bboxes[-1]
           track_queries['track_dec_out_logits'] = track_dec_out_logits[-1]
           
           track_queries['track_dec_out_bboxes_b'] = track_dec_out_bboxes[-1].clone()
           track_queries['track_dec_out_logits_b'] = track_dec_out_logits[-1].clone()
        """#comentado por que no se usa
        if mode == 0:
          track_queries['hs'] = outputs
        else:
          #track_queries['hs_2'] = outputs #probar
          #if 'embeds_2' in track_queries and track_queries['embeds_2'] is not None:
          #  track_queries['embeds_2_b'] = track_queries['embeds_2'].clone()
          track_queries['embeds_2'] = outputs
        return dec_out_bboxes, dec_out_logits, track_queries

class TransformerDecoderMotTrack(nn.Module):
      def __init__(self, hidden_dim, decoder_layer, num_layers):
          super(TransformerDecoderMotTrack, self).__init__()
          self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
      def forward(self,tgt, memory, query_pos_head_cons, boxes):
          query_pos_embed = query_pos_head_cons(boxes)
          for i, layer in enumerate(self.layers):
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
        self.keep_add_head = MLP(hidden_dim, 2 * hidden_dim, 2, num_layers=6)
        #self.proj_cons = nn.Linear(hidden_dim,hidden_dim)
        self.proj_cons = MLP(hidden_dim, 2 * hidden_dim, hidden_dim, num_layers=6)
        self.proj_boxes = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=6)

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
            track_queries=track_queries,
            mode=0)
        
        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            _ , track_queries['hs'] = torch.split(track_queries['hs'], dn_meta['dn_num_split'], dim=1)
        

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        
        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        th = 0.5
        with torch.no_grad():
             scores = F.sigmoid(out_logits[-1])
             mask = torch.max(scores,2)[0] > th

        #decoder 2 Prueba
        if 'embeds_2' in track_queries and track_queries['embeds_2'] is not None and ind_track:
          #if not self.training:
            #print('entra a actualizar')
          out_bboxes_2, out_logits_2, track_queries = self.decoder(
            track_queries['embeds_2'],
            inverse_sigmoid(track_queries['pred_boxes_2']),
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            0,
            attn_mask=None,
            track_queries=track_queries,
            mode=1)
          #track_queries['embeds_2'] = track_queries['hs_2']
          track_queries['pred_boxes_2'] = out_bboxes_2[-1]
          track_queries['pred_logits_2'] = out_logits_2[-1]
          #track_queries['pred_boxes_2_b'] = out_bboxes_2[-1].clone()
          #track_queries['pred_logits_2_b'] = out_logits_2[-1].clone()

          #track_queries = self.qim_int(track_queries) # Probando qim module antes de la asociacion con los dets
          t = time.localtime()
          current_time = time.strftime("%H_%M_%S", t)
          #np.save('embeds_2_'+current_time,track_queries['embeds_2'].cpu().detach().numpy())
          #np.save('boxes_2_'+current_time,track_queries['pred_boxes_2'].cpu().detach().numpy())


        if mask.sum() > 0 and ind_track:
          t = time.localtime()
          current_time = time.strftime("%H_%M_%S", t)
          #np.save('hs_'+current_time,track_queries['hs'][mask.clone()].cpu().detach().numpy())
          #np.save('hs_boxes_'+current_time,out_bboxes[-1][mask].cpu().detach().numpy())
          #aux_track_queries_tgt, attn_weights, attn_weights_logits = self.track_decoder_layer_h(track_queries["embeds_2"],None,track_queries['hs'],None,None,None,None,self.query_pos_head(inverse_sigmoid(track_queries["pred_boxes_2"].detach().clone())), return_attn=True)
          """comentado la parte del transformer entre consultas
          if 'embeds_2' in track_queries and track_queries['embeds_2'] is not None:
            #aux_track_queries_tgt, attn_weights, attn_weights_logits = self.track_decoder_layer_h(track_queries["embeds_2"],None,track_queries['hs'][mask].unsqueeze(0),None,None,None,None,self.query_pos_head(inverse_sigmoid(track_queries["pred_boxes_2"].clone())), return_attn=True)
              aux_hs = track_queries['hs'][mask.clone()]
              aux_hs = aux_hs.unsqueeze(0)
              aux_track_queries_tgt, attn_weights, attn_weights_logits = self.track_decoder_layer_h(track_queries["embeds_2"],None,aux_hs,None,None,None,None,self.query_pos_head(inverse_sigmoid(track_queries["pred_boxes_2"])), return_attn=True)
              attn_weights = attn_weights.mean(0)
              attn_weights_logits = attn_weights_logits.mean(0)
              track_queries["embeds_2"] = aux_track_queries_tgt #probando actualizando los trackqueries despues de ver los det queries
              track_queries['attn_weights'] = attn_weights_logits
              track_queries["det_idxs"] = self.det_id_h(aux_track_queries_tgt.squeeze(0)) # comentado por ahora ya que aun no lo uso
              #Probando actualizar las cajas mediante los nuevos outputs
              track_queries['pred_boxes_2'] = F.sigmoid(self.dec_bbox_head[-1](track_queries["embeds_2"]) + inverse_sigmoid(track_queries['pred_boxes_2'])) #probar con cuidado
              track_queries['pred_boxes_2_b'] = track_queries['pred_boxes_2'].clone()
              track_queries['pred_logits_2'] = self.dec_score_head[-1](track_queries["embeds_2"])
              track_queries['pred_logits_2_b'] = track_queries['pred_logits_2'].clone()
          """
          #track_queries['cons_det'] = self.proj_cons(track_queries['hs'][mask.clone()])
          track_queries['cons_det'] = self.proj_cons(track_queries['hs'][mask.clone()]+self.proj_boxes(out_bboxes[-1][mask.clone()]))
          if 'embeds_2' not in track_queries:
              track_queries['embeds_2'] = track_queries['hs'][mask].unsqueeze(0)#.clone()
              track_queries['pred_boxes_2'] = out_bboxes[-1][mask].unsqueeze(0)#.clone()
              track_queries['pred_logits_2'] = out_logits[-1][mask].unsqueeze(0)#.clone()
              #track_queries['cons_track'] = self.proj_cons(track_queries['embeds_2'].squeeze(0))
              track_queries['cons_track'] = self.proj_cons(track_queries['embeds_2'].squeeze(0)+self.proj_boxes(track_queries['pred_boxes_2'].squeeze(0)))
              #print('shape constrack 1:',track_queries['cons_track'].shape)
              t = time.localtime()
              current_time = time.strftime("%H_%M_%S", t)
              #np.save('embeds_2_'+current_time,track_queries['embeds_2'].cpu().detach().numpy())
              #np.save('boxes_2_'+current_time,track_queries['pred_boxes_2'].cpu().detach().numpy())
              if "id_max" not in track_queries:
                 track_queries['ids'] = torch.arange(0,track_queries['embeds_2'].shape[1]).to(track_queries['embeds_2'].device)
              track_queries['id_max'] = torch.max(track_queries["ids"])
              track_queries['delete'] = torch.zeros(track_queries['embeds_2'].shape[1]).to(track_queries['embeds_2'].device)
          else:
            #Probando con la interaccion de las consultas de seguimiento y deteccion
            #aux_track_queries_tgt, attn_weights, attn_weights_logits = self.track_decoder_layer_h(track_queries["embeds_2"],None,track_queries['hs'][mask.clone()].unsqueeze(0),None,None,None,None,self.query_pos_head(inverse_sigmoid(track_queries["pred_boxes_2"].clone())), return_attn=True)
            #track_queries["embeds_2"] = aux_track_queries_tgt
            #track_queries['pred_boxes_2'] = F.sigmoid(self.dec_bbox_head[-1](track_queries["embeds_2"]) + inverse_sigmoid(track_queries['pred_boxes_2'])) #probar con cuidado
            #track_queries['pred_logits_2'] = self.dec_score_head[-1](track_queries["embeds_2"])
            #Probando con la interaccion de las consultas de seguimiento y deteccion
            #Pruebas de similitud
            #track_queries['cons_track'] = self.proj_cons(track_queries['embeds_2'].squeeze(0))
            track_queries['cons_track'] = self.proj_cons(track_queries['embeds_2'].squeeze(0)+self.proj_boxes(track_queries['pred_boxes_2'].squeeze(0)))
            #print('shape constrack 2:',track_queries['cons_track'].shape)
            res = F.normalize(track_queries['cons_track'] @ F.normalize(track_queries['cons_det']).t())
            ious = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['pred_boxes_2'].squeeze(0)),box_cxcywh_to_xyxy(out_bboxes[-1][mask.clone()]))
            
           
            track_queries['embeds_2_b'] = track_queries['embeds_2'].clone()
            track_queries['pred_logits_2_b'] = track_queries['pred_logits_2'].clone()
            track_queries['pred_boxes_2_b'] = track_queries['pred_boxes_2'].clone()

            #th_assoc = 0.1
            #sim_idxs = torch.nonzero(res > th_assoc)[:,1]
            #mask[:,torch.unique(torch.nonzero(mask[0])[sim_idxs])] = False

            #Agregar nuevas queries mediante los ious solo en entrenamiento
            th_assoc = 0.5
            if self.training:
                sim_idxs = torch.nonzero(ious > th_assoc)[:,1]
                sim_del = torch.nonzero(ious < th_assoc)[:,0]
            else:
                sim_idxs = torch.nonzero(res > th_assoc)[:,1]
                sim_del = torch.nonzero(res < th_assoc)[:,0]
            mask[:,torch.unique(torch.nonzero(mask[0])[sim_idxs])] = False

            if not self.training:
                print('ious:',ious)
                print('res:',res)

            track_queries['delete'][torch.unique(sim_del)] +=1

            track_queries['embeds_2'] = torch.cat((track_queries['embeds_2'],track_queries['hs'][mask.clone()].unsqueeze(0)),dim=1)
            track_queries['pred_logits_2'] = torch.cat((track_queries['pred_logits_2'],out_logits[-1][mask.clone()].unsqueeze(0)),dim=1)
            track_queries['pred_boxes_2'] = torch.cat((track_queries['pred_boxes_2'],out_bboxes[-1][mask.clone()].unsqueeze(0)),dim=1)
            track_queries['delete'] = torch.cat((track_queries['delete'],torch.zeros(track_queries['hs'][mask.clone()].shape[0]).to(track_queries['embeds_2'].device)),0)
            
            if "id_max" not in track_queries:
                track_queries['ids'] = torch.arange(0,track_queries['embeds_2'].shape[1]).to(track_queries['embeds_2'].device)
            else:
                id_max = track_queries['id_max']
                track_queries["ids"] = torch.cat((track_queries["ids"],torch.arange(id_max+1,id_max+1+mask.sum()).to(track_queries["embeds_2"].device)))
            track_queries['id_max'] = torch.max(track_queries["ids"])
            

          """"
          else:
            #track_queries = self.qim_int(track_queries) #qim module
            th_assoc = 0.2
            attn_idxs = torch.nonzero(track_queries['attn_weights'] > th_assoc)[:,1]

            r_add = self.keep_add_head(track_queries['embeds_2'])
            with torch.no_grad():
                print('softmax r_add:',torch.softmax(r_add,dim=-1))
            
            if not self.training:
                print('attn weights:',track_queries['attn_weights'])
                #print('attn weights sigmoid:',torch.sigmoid(track_queries['attn_weights']))
                #print('attn weights softmax:',F.softmax(track_queries['attn_weights'],dim=1))
                print('attn idxs:',attn_idxs)
                print('idxs det:',torch.nonzero(mask[0]))
                print('indices reales a utilizar:',torch.unique(torch.nonzero(mask[0])[attn_idxs]))
                print('como van los ids:',track_queries['ids'])

            mask[:,torch.unique(torch.nonzero(mask[0])[attn_idxs])] = False #probando el evitar detecciones duplicadas
            if not self.training:
                if mask.sum()>0:
                    print('va a agregar nueva deteccion')
            track_queries['embeds_2'] = torch.cat((track_queries['embeds_2'],track_queries['hs'][mask].unsqueeze(0)),dim=1)
            track_queries['pred_logits_2'] = torch.cat((track_queries['pred_logits_2'],out_logits[-1][mask].unsqueeze(0)),dim=1)
            track_queries['pred_boxes_2'] = torch.cat((track_queries['pred_boxes_2'],out_bboxes[-1][mask].unsqueeze(0)),dim=1)

            if "id_max" not in track_queries:
                track_queries['ids'] = torch.arange(0,track_queries['embeds_2'].shape[1]).to(track_queries['embeds_2'].device)
            else:
                id_max = track_queries['id_max']
                track_queries["ids"] = torch.cat((track_queries["ids"],torch.arange(id_max+1,id_max+1+mask.sum()).to(track_queries["embeds_2"].device)))
            track_queries['id_max'] = torch.max(track_queries["ids"])
        """#comentado para probar
        """
        if not self.training and ind_track:
            if 'embeds_2' in track_queries:
                print('como van las track_queries:',track_queries['embeds_2'].shape)
                print('boxes:',track_queries['pred_boxes_2'])
                print('ids:',track_queries['ids'])
                with torch.no_grad():
                    print('scores:',torch.sigmoid(track_queries['pred_logits_2']))
        """
        return out, track_queries
