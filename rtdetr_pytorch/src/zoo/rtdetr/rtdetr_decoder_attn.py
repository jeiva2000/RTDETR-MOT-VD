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


from src.core import register


__all__ = ['RTDETRTransformer']


def sinusoidal_positional_embedding(token_sequence_size, token_embedding_dim, n=10000.0):
    if token_embedding_dim % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))

    T = token_sequence_size
    d = token_embedding_dim #d_model=head_num*d_k, not d_q, d_k, d_v

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings

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
         value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
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


class TransformerDecoderLayer(nn.Module):
   def __init__(self,
                d_model=256,
                n_head=8,
                dim_feedforward=1024,
                dropout=0.,
                activation="relu",
                n_levels=4,
                n_points=4,):

      super(TransformerDecoderLayer, self).__init__()

      # self attention
      self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
      self.dropout1 = nn.Dropout(dropout)
      self.norm1 = nn.LayerNorm(d_model)

      # cross attention
      self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
      self.dropout2 = nn.Dropout(dropout)
      self.norm2 = nn.LayerNorm(d_model)

      # ffn
      self.linear1 = nn.Linear(d_model, dim_feedforward)
      self.activation = getattr(F, activation)
      self.dropout3 = nn.Dropout(dropout)
      self.linear2 = nn.Linear(dim_feedforward, d_model)
      self.dropout4 = nn.Dropout(dropout)
      self.norm3 = nn.LayerNorm(d_model)

   def with_pos_embed(self, tensor, pos):
      return tensor if pos is None else tensor + pos

   def forward_ffn(self, tgt):
      return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))


   def forward(self,
             tgt,
             reference_points,
             memory,
             memory_spatial_shapes,
             memory_level_start_index,
             attn_mask=None,
             memory_mask=None,
             query_pos_embed=None):
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

      return tgt, sampling_values


class TransformerDecoder(nn.Module):
   def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):

      super(TransformerDecoder, self).__init__()
      self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
      self.hidden_dim = hidden_dim
      self.num_layers = num_layers
      self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

   def forward(self,
             tgt,
             ref_points_unact,
             memory,
             memory_spatial_shapes,
             memory_level_start_index,
             bbox_head,
             score_head,
             query_pos_head,
             attn_mask=None,
             memory_mask=None):
      output = tgt
      dec_out_bboxes = []
      dec_out_logits = []
      ref_points_detach = F.sigmoid(ref_points_unact)

      for i, layer in enumerate(self.layers):
         ref_points_input = ref_points_detach.unsqueeze(2)
         query_pos_embed = query_pos_head(ref_points_detach)

         output, _ = layer(output, ref_points_input, memory,
                        memory_spatial_shapes, memory_level_start_index,
                        attn_mask, memory_mask, query_pos_embed)

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

      return output, torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)

#Transformer block
class TransformerBlock(nn.Module):
   def __init__(self, embed_size, heads, dropout, forward_expansion):
      super(TransformerBlock, self).__init__()
      self.Wq = nn.Linear(embed_size, embed_size)
      self.Wk = nn.Linear(embed_size, embed_size)
      self.Wv = nn.Linear(embed_size, embed_size)

      self.ff_1 = nn.Linear(embed_size, forward_expansion * embed_size)
      self.ff_2 = nn.Linear(forward_expansion * embed_size, embed_size)

      self.norm1 = nn.LayerNorm(embed_size)
      self.norm2 = nn.LayerNorm(embed_size)

      self.dropout = nn.Dropout(dropout)#self._reset_parameters()

   def _reset_parameters(self):
       # Original Transformer initialization, see PyTorch documentation
       nn.init.xavier_uniform_(self.Wq.weight)
       nn.init.xavier_uniform_(self.Wk.weight)
       nn.init.xavier_uniform_(self.Wv.weight)
       self.Wq.bias.data.fill_(0)
       self.Wq.bias.data.fill_(0)
       self.Wq.bias.data.fill_(0)
       nn.init.xavier_uniform_(self.ff_1.weight)
       nn.init.xavier_uniform_(self.ff_2.weight)
       self.ff_1.bias.data.fill_(0)
       self.ff_2.bias.data.fill_(0)

   def attention(self, v, k, q, mask=None):
      attn_logits = torch.matmul(q, k.transpose(-2, -1))
      attn_logits = attn_logits / (k.size(-1) ** 0.5)
      if mask is not None:
         attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
      attention = F.sigmoid(attn_logits)
      values = torch.matmul(attention, v)
      return values, attention

   def feed_forward(self, x):
      return self.ff_2(F.relu(self.ff_1(x)))

   def forward(self, value, key, query, split=None, embed0=None, embed1=None, mask=None, embed=None):
      """
      if split is not None:
         if embed0 is not None:
            query[:,:split] = query[:,:split] + embed0
            key[:,:split] = key[:,:split] + embed0
            value[:,:split] = value[:,:split] + embed0

         if embed1 is not None:
            query[:,split:] = query[:,split:] + embed1
            key[:,split:] = key[:,split:] + embed1
            value[:,split:] = value[:,split:] + embed1
      else:
         query = query + embed0
         key = key + embed0
         value = value + embed0
      """
      
      query = query + embed0
      key = key + embed0
      value = value + embed0
      q, k, v = self.Wq(query), self.Wk(key), self.Wv(value)
      x, attn = self.attention(v, k, q, mask)
      x = self.norm1(x)
      x = self.feed_forward(x)
      out = self.norm2(x)

      return out, attn

#Decoder track
class TransformerDecoderTrack(nn.Module):
   def __init__(self, embed_size, heads, dropout, forward_expansion, num_layers, seq_len):
      super(TransformerDecoderTrack, self).__init__()
      self.layers = nn.ModuleList(
         [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
         )
      #self.embeddings = nn.Embedding(seq_len, embed_size)
      #self.emb_proj_0 = MLP(embed_size, 2 * embed_size, embed_size, num_layers=3)
      #self.emb_proj_1 = MLP(embed_size, 2 * embed_size, embed_size, num_layers=3)      
      self.embed_size = embed_size

   def forward(self, x, pos_2d=None, pos_temp= None, split=None, mask=None):
      """
      if split is not None:
         emb0 = self.emb_proj_0(x[:,:split])
         emb1 = self.emb_proj_1(x[:,split:])
      """
      #if split is not None:
         #emb0 = self.emb_proj_0(x[:,:split])
      x = x + pos_temp
      for layer in self.layers:
         x, attn = layer(x,x,x, split=split, embed0=pos_2d, embed1=pos_temp,mask=mask)
      return x, attn

@register
class RTDETRTransformer(nn.Module):
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

      super(RTDETRTransformer, self).__init__()
      assert position_embed_type in ['sine', 'learned'], \
         f'ValueError: position_embed_type not supported {position_embed_type}!'
      assert len(feat_channels) <= num_levels
      assert len(feat_strides) == len(feat_channels)
      for _ in range(num_levels - len(feat_strides)):
         feat_strides.append(feat_strides[-1] * 2)

      self.hidden_dim = hidden_dim
      self.nhead = nhead
      self.feat_strides = feat_strides
      self.num_levels = num_levels
      self.num_classes = num_classes
      self.num_queries = num_queries
      self.eps = eps
      self.num_decoder_layers = num_decoder_layers
      self.eval_spatial_size = eval_spatial_size
      self.aux_loss = aux_loss

      # backbone feature projection
      self._build_input_proj_layer(feat_channels)

      # Transformer module
      decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
      self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

      self.num_denoising = num_denoising
      self.label_noise_ratio = label_noise_ratio
      self.box_noise_scale = box_noise_scale
      # denoising part
      if num_denoising > 0: 
         # self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim, padding_idx=num_classes-1) # TODO for load paddle weights
         self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)

      # decoder embedding
      self.learnt_init_query = learnt_init_query
      if learnt_init_query:
         self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
      self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

      # encoder head
      self.enc_output = nn.Sequential(
         nn.Linear(hidden_dim, hidden_dim),
         nn.LayerNorm(hidden_dim,)
      )
      self.enc_score_head = nn.Linear(hidden_dim, num_classes)
      self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

      # decoder head
      self.dec_score_head = nn.ModuleList([
         nn.Linear(hidden_dim, num_classes)
         for _ in range(num_decoder_layers)
      ])
      self.dec_bbox_head = nn.ModuleList([
         MLP(hidden_dim, hidden_dim, 4, num_layers=3)
         for _ in range(num_decoder_layers)
      ])

      # init encoder output anchors and valid_mask
      if self.eval_spatial_size:
         self.anchors, self.valid_mask = self._generate_anchors()

      #track
      """
      self.summary = False
      if self.summary:
         hidden_dim_track = hidden_dim*2
         self.summary_proj = MLP(hidden_dim_track, hidden_dim_track, hidden_dim, num_layers=3)
      else:
         hidden_dim_track = hidden_dim
      """
      self.use_embs_id = True
      self.use_hist = True
      if self.use_embs_id:
         hidden_dim_track = 512
         self.query_pos_head_track = MLP(4, 2 * hidden_dim_track, hidden_dim_track, num_layers=2)
      else:
         hidden_dim_track = 256

      if self.use_hist:
         #proj emb id with history
         self.embs_id = MLP(hidden_dim*5,hidden_dim*3,hidden_dim,num_layers=3)
      else:
         self.embs_id = nn.Embedding(8000,hidden_dim)
      
      self.dec_track = TransformerDecoderTrack(embed_size=hidden_dim_track, heads=nhead, dropout=0.1, forward_expansion=4, num_layers=6, seq_len=100)
      self.temporal_encoder = sinusoidal_positional_embedding(5000,hidden_dim_track) #nn.Embedding(2000,hidden_dim_track) #
      #self.temporal_encoder = nn.Embedding(2000,hidden_dim_track)
      self._reset_parameters()

   def _reset_parameters(self):
      bias = bias_init_with_prob(0.01)

      init.constant_(self.enc_score_head.bias, bias)
      init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
      init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

      for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
         init.constant_(cls_.bias, bias)
         init.constant_(reg_.layers[-1].weight, 0)
         init.constant_(reg_.layers[-1].bias, 0)
      
      # linear_init_(self.enc_output[0])
      init.xavier_uniform_(self.enc_output[0].weight)
      if self.learnt_init_query:
         init.xavier_uniform_(self.tgt_embed.weight)
      init.xavier_uniform_(self.query_pos_head.layers[0].weight)
      init.xavier_uniform_(self.query_pos_head.layers[1].weight)

   def _build_input_proj_layer(self, feat_channels):
      self.input_proj = nn.ModuleList()
      for in_channels in feat_channels:
         self.input_proj.append(
             nn.Sequential(OrderedDict([
                  ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                  ('norm', nn.BatchNorm2d(self.hidden_dim,))])
             )
         )

      in_channels = feat_channels[-1]

      for _ in range(self.num_levels - len(feat_channels)):
         self.input_proj.append(
             nn.Sequential(OrderedDict([
                  ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                  ('norm', nn.BatchNorm2d(self.hidden_dim))])
             )
         )
         in_channels = self.hidden_dim

   def _get_encoder_input(self, feats):
      # get projection features
      proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
      if self.num_levels > len(proj_feats):
         len_srcs = len(proj_feats)
         for i in range(len_srcs, self.num_levels):
             if i == len_srcs:
                  proj_feats.append(self.input_proj[i](feats[-1]))
             else:
                  proj_feats.append(self.input_proj[i](proj_feats[-1]))

      # get encoder inputs
      feat_flatten = []
      spatial_shapes = []
      level_start_index = [0, ]
      for i, feat in enumerate(proj_feats):
         _, _, h, w = feat.shape
         # [b, c, h, w] -> [b, h*w, c]
         feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
         # [num_levels, 2]
         spatial_shapes.append([h, w])
         # [l], start index of each level
         level_start_index.append(h * w + level_start_index[-1])

      # [b, l, c]
      feat_flatten = torch.concat(feat_flatten, 1)
      level_start_index.pop()
      return (feat_flatten, spatial_shapes, level_start_index)


   def _generate_anchors(self,
                        spatial_shapes=None,
                        grid_size=0.05,
                        dtype=torch.float32,
                        device='cpu'):
      if spatial_shapes is None:
         spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
             for s in self.feat_strides
         ]
      anchors = []
      for lvl, (h, w) in enumerate(spatial_shapes):
         grid_y, grid_x = torch.meshgrid(\
             torch.arange(end=h, dtype=dtype), \
             torch.arange(end=w, dtype=dtype), indexing='ij')
         grid_xy = torch.stack([grid_x, grid_y], -1)
         valid_WH = torch.tensor([w, h]).to(dtype)
         grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
         wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
         anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

      anchors = torch.concat(anchors, 1).to(device)
      valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
      anchors = torch.log(anchors / (1 - anchors))
      # anchors = torch.where(valid_mask, anchors, float('inf'))
      # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
      anchors = torch.where(valid_mask, anchors, torch.inf)

      return anchors, valid_mask


   def _get_decoder_input(self,
                        memory,
                        spatial_shapes,
                        denoising_class=None,
                        denoising_bbox_unact=None):
      bs, _, _ = memory.shape
      # prepare input for decoder
      if self.training or self.eval_spatial_size is None:
         anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
      else:
         anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

      # memory = torch.where(valid_mask, memory, 0)
      memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export 

      output_memory = self.enc_output(memory)

      enc_outputs_class = self.enc_score_head(output_memory)
      enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

      _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
      
      reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
         index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))

      enc_topk_bboxes = F.sigmoid(reference_points_unact)
      if denoising_bbox_unact is not None:
         reference_points_unact = torch.concat(
             [denoising_bbox_unact, reference_points_unact], 1)
      
      enc_topk_logits = enc_outputs_class.gather(dim=1, \
         index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

      # extract region features
      if self.learnt_init_query:
         target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
      else:
         target = output_memory.gather(dim=1, \
             index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
         target = target.detach()

      if denoising_class is not None:
         target = torch.concat([denoising_class, target], 1)

      return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits


   def forward(self, feats, targets=None, track_queries=None, ind_track=True, criterion=None, mode='', limit=3, update_track=False, th=0.5, n_frame=None):
      #print('como llegan las targets:',targets)
      #if self.training:
      #   print('revisar boxes shape:',targets[0]['boxes'].shape)
      #if targets[0]['boxes'].shape[0] < 6:
      #   print('target:',targets)
      device = feats[0].device
      # input projection and embedding
      (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)
      #print("spatial_shapes:",spatial_shapes)
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
      
      
      if 't_embs' in track_queries:
         aux_target = target.clone().detach()
         target = torch.cat((target,track_queries['t_embs']),dim=1)
         init_ref_points_unact = torch.cat((init_ref_points_unact,inverse_sigmoid(track_queries['boxes'])),dim=1)

         #mask
         if attn_mask is None:
            attn_mask = torch.full([track_queries['t_embs'].shape[1]+aux_target.shape[1], track_queries['t_embs'].shape[1]+aux_target.shape[1]], False, dtype=torch.bool, device=device)
            attn_mask[:aux_target.shape[1],aux_target.shape[1]:] = True 
            attn_mask[aux_target.shape[1]:,:aux_target.shape[1]] = True
         else:
            attn_mask_0 = torch.full((aux_target.shape[1],track_queries['t_embs'].shape[1]),True,dtype=torch.bool, device=device)
            attn_mask_1 = torch.full((track_queries['t_embs'].shape[1],aux_target.shape[1]+track_queries['t_embs'].shape[1]),True,dtype=torch.bool, device=device)
            attn_mask_1[:,aux_target.shape[1]:]=False
            attn_mask = torch.cat((attn_mask,attn_mask_0),dim=1)
            attn_mask = torch.cat((attn_mask,attn_mask_1),dim=0)

            dn_meta['dn_num_split'][1]+=track_queries['t_embs'].shape[1]

         #evitando la autoatencion
         eye_d = torch.eye(track_queries['t_embs'].shape[1],track_queries['t_embs'].shape[1]).to(device)
         eye_d = eye_d>0
         attn_mask[aux_target.shape[1]:,aux_target.shape[1]:] = ~eye_d
      

      # decoder
      output, out_bboxes, out_logits = self.decoder(
         target,
         init_ref_points_unact,
         memory,
         spatial_shapes,
         level_start_index,
         self.dec_bbox_head,
         self.dec_score_head,
         self.query_pos_head,
         attn_mask=attn_mask)

      
      if dn_meta is not None:
         _, output = torch.split(output, dn_meta['dn_num_split'], dim=1)
         dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
         dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
      
      
      #split tracks and dets
      if 't_embs' in track_queries:
         output, track_queries['t_embs'] = torch.split(output, [self.num_queries,track_queries['t_embs'].shape[1]], dim=1)
         out_bboxes, track_queries['boxes'] = torch.split(out_bboxes, [self.num_queries,track_queries['t_embs'].shape[1]], dim=2)
         out_logits, track_queries['logits'] = torch.split(out_logits, [self.num_queries,track_queries['t_embs'].shape[1]], dim=2)
         track_queries['boxes'] = track_queries['boxes'][-1]
         track_queries['logits'] = track_queries['logits'][-1]

      out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

      if self.training and self.aux_loss:
         out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
         out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))
         
         if self.training and dn_meta is not None:
            out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
            out['dn_meta'] = dn_meta

      if self.training:
         outputs_without_aux = {k: v for k, v in out.items() if 'aux' not in k}
         indices = criterion.matcher(outputs_without_aux, targets)
         idxs_m = indices[0][0]
         idxs_m = idxs_m.to(device)
         idxs = idxs_m
         out['gts'] = targets[0]['ids'][indices[0][1]]
         out['gts'] = out['gts'].to(device)
         #idxs_f = None
         idxs_f = idxs
         
      else:
         values, _= torch.max(F.sigmoid(out_logits[-1]),2)
         values = values.squeeze(0)
         idxs_f = torch.nonzero(values>0.5).flatten()

      aux_output = output.clone()
      if idxs_f is not None:
         aux_output = aux_output[:,idxs_f]

      out['det_out'] = aux_output
      
      if self.use_embs_id:
         if not self.use_hist:
            aux_pos_out = self.embs_id(torch.zeros(len(idxs_f)).to(device).long()).unsqueeze(0).to(device)
         else:
            #hist
            aux_pos_out = torch.repeat_interleave(aux_output, 5, dim=1).reshape(len(idxs_f),5,256).reshape(len(idxs_f),5*256).unsqueeze(0)
            aux_pos_out = self.embs_id(aux_pos_out)

      if 't_embs' in track_queries:

         new_outputs = torch.cat((aux_output,track_queries['t_embs']), dim=1)
         new_boxes = torch.cat((out['pred_boxes'][:,idxs_f],track_queries['boxes']), dim=1)

         if self.use_embs_id:
            new_outputs = torch.cat((new_outputs,torch.cat((aux_pos_out,track_queries['pos_emb']),dim=1)),dim=2)
            t_embs, attn = self.dec_track(new_outputs, pos_2d=self.query_pos_head_track(new_boxes.clone().detach()), pos_temp=self.temporal_encoder[n_frame].to(device))
         else:
            t_embs, attn = self.dec_track(new_outputs, pos_2d=self.query_pos_head(new_boxes.clone().detach()), pos_temp=self.temporal_encoder[n_frame].to(device))

         track_queries['t_embs'] = t_embs[:,aux_output.shape[1]:,:self.hidden_dim]

         #hist
         if 'hist_embs' not in track_queries:
            track_queries['hist_embs'] = [track_queries['t_embs'][:,j] for j in range(track_queries['t_embs'].shape[1])]
         else:
            track_queries['hist_embs'] = [torch.cat((track_queries['hist_embs'][j],track_queries['t_embs'][:,j]),dim=0) for j in range(len(track_queries['hist_embs']))]
            track_queries['hist_embs'] = [hist[-5:] if hist.shape[0]>5 else hist for hist in track_queries['hist_embs']]

         #Update pos emb by hist -- poner mejor despues
         if self.use_hist:
            for j in range(len(track_queries['hist_embs'])):
               aux_pos = track_queries['hist_embs'][j].unsqueeze(0)
               if track_queries['hist_embs'][j].shape[0] < 5:
                  aux_r = 5 - track_queries['hist_embs'][j].shape[0]
                  aux_pos = track_queries['hist_embs'][j][0].repeat(aux_r, 1).unsqueeze(0)#.reshape(1,1,5*256)
                  aux_pos = torch.cat((track_queries['hist_embs'][j].unsqueeze(0),aux_pos),dim=1).reshape(1,1,5*256)
               else:
                  aux_pos = aux_pos.reshape(1,1,5*256)
            track_queries['pos_emb'][:,j] = self.embs_id(aux_pos)

      else:
         if self.use_embs_id:
            _, attn = self.dec_track(torch.cat((aux_output,aux_pos_out),dim=2), pos_2d=self.query_pos_head_track(out['pred_boxes'][:,idxs_f].clone().detach()), pos_temp=self.temporal_encoder[n_frame].to(device))
         else:
            _, attn = self.dec_track(aux_output, pos_2d=self.query_pos_head(out['pred_boxes'][:,idxs_f].clone().detach()), pos_temp=self.temporal_encoder[n_frame].to(device))


      out['attn'] = attn.squeeze(0)

      if self.training and 't_embs' in track_queries:
         out['gts'] = torch.cat((out['gts'],track_queries['ids_gt']))

      if self.training:
         losses_dict, track_queries = criterion(out, targets, track_queries)

      if 'ids_gt' not in track_queries and self.training:
         track_queries['ids_gt'] = targets[0]['ids'][indices[0][1]]
         track_queries['t_embs'] = output[:,indices[0][0]]
         track_queries['boxes'] = out_bboxes[-1][:,indices[0][0]]
         track_queries['logits'] = out_logits[-1][:,indices[0][0]]
         if self.use_embs_id:
            if not self.use_hist:
               track_queries['pos_emb'] = self.embs_id(torch.arange(1,len(indices[0][1])+1).to(device).long()).unsqueeze(0).to(device)
            else:
               #hist pos emb
               aux_pos = torch.repeat_interleave(track_queries['t_embs'], 5, dim=1).reshape(len(idxs_f),5,256).reshape(len(idxs_f),5*256)
               track_queries['pos_emb'] = self.embs_id(aux_pos).unsqueeze(0)

      elif self.training:
         aux_idxs = torch.nonzero(torch.isin(targets[0]['ids'][indices[0][1]],track_queries['ids_gt'],invert=True)).flatten()
         if len(aux_idxs) > 0:
            track_queries['ids_gt'] = torch.cat((track_queries['ids_gt'],targets[0]['ids'][aux_idxs]))
            track_shape = track_queries['t_embs'].shape[1]
            for i, idx in enumerate(indices[0][1]):
               if idx in aux_idxs:
                  track_queries['t_embs'] = torch.cat((track_queries['t_embs'],output[:,indices[0][0][i]].unsqueeze(0)),dim=1)
                  track_queries['boxes'] = torch.cat((track_queries['boxes'],out_bboxes[-1][:,indices[0][0][i]].unsqueeze(0)),dim=1)
                  track_queries['logits'] = torch.cat((track_queries['logits'],out_logits[-1][:,indices[0][0][i]].unsqueeze(0)),dim=1)
                  #aux_mask = torch.nonzero(torch.isin(idxs_m,indices[0][0][i].to(device)))
                  #track_queries['hist_embs'].append(output[:,indices[0][0][i]])
                  
                  if self.use_hist:
                     aux_pos = torch.repeat_interleave(output[:,indices[0][0][i]], 5, dim=0).reshape(5*self.hidden_dim).unsqueeze(0)
                     aux_pos = self.embs_id(aux_pos).unsqueeze(0)
                     track_queries['pos_emb'] = torch.cat((track_queries['pos_emb'],aux_pos),dim=1)
                  
            assert track_queries['ids_gt'].shape[0] == track_queries['t_embs'].shape[1]

            if self.use_embs_id and not self.use_hist:
               aux_shape = track_queries['t_embs'].shape[1] - track_shape
               if aux_shape > 0:
                  aux_pos = self.embs_id(torch.arange(track_shape,track_shape+aux_shape).to(device).long()).unsqueeze(0).to(device)
                  track_queries['pos_emb'] = torch.cat((track_queries['pos_emb'],aux_pos),dim=1)

      if not self.training:
         if 't_embs' not in track_queries:
            if len(idxs_f) > 0:
               track_queries['t_embs'] = output[:,idxs_f]
               track_queries['boxes'] = out_bboxes[-1][:,idxs_f]
               track_queries['logits'] = out_logits[-1][:,idxs_f]
               track_queries['delete'] = torch.zeros(track_queries['t_embs'].shape[1]).to(device)
               track_queries['ids'] = torch.arange(0,track_queries['t_embs'].shape[1])
               if self.use_embs_id:
                  if not self.use_hist:
                     track_queries['pos_emb'] = self.embs_id(torch.arange(1,len(idxs_f)+1).to(device).long()).unsqueeze(0).to(device)
                  else:
                     aux_pos = torch.repeat_interleave(track_queries['t_embs'], 5, dim=1).reshape(len(idxs_f),5,256).reshape(len(idxs_f),5*256)
                     track_queries['pos_emb'] = self.embs_id(aux_pos).unsqueeze(0)
                  
         else:
            track_queries['boxes_det'] = out_bboxes[-1][:,idxs_f]
            track_queries['logits_det'] = out_logits[-1][:,idxs_f]
            track_shape = track_queries['t_embs'].shape[1]
            det_shape = output[:,idxs_f].shape[1]
            values = torch.max(out['attn'][:det_shape,det_shape:],1)[0]
            #aux_idxs = idxs_f[values<0.5]
            aux_idxs = idxs_f[values==0]
            #aux_idxs_2 = torch.nonzero(values<0.5).flatten()
            track_shape = track_queries['t_embs'].shape[1]

            if track_shape+len(aux_idxs) > 8000: #evitar mas de 8000 tracks
               aux_idxs = torch.tensor([]).to(device)
            #print('t_embs antes:',track_queries['t_embs'].shape)
            track_queries['t_embs'] = torch.cat((track_queries['t_embs'],output[:,aux_idxs]),dim=1)
            track_queries['boxes'] = torch.cat((track_queries['boxes'],out_bboxes[-1][:,aux_idxs]),dim=1)
            track_queries['logits'] = torch.cat((track_queries['logits'],out_logits[-1][:,aux_idxs]),dim=1)
            track_queries['delete'] = torch.cat((track_queries['delete'],torch.zeros(len(aux_idxs)).to(device)))
            
            if self.use_embs_id:
               aux_shape = track_queries['t_embs'].shape[1] - track_shape
               if aux_shape > 0:
                  if not self.use_hist:
                     aux_pos = self.embs_id(torch.arange(track_shape,track_shape+aux_shape).to(device).long()).unsqueeze(0).to(device)
                     track_queries['pos_emb'] = torch.cat((track_queries['pos_emb'],aux_pos),dim=1)
                  else:
                     aux_pos = torch.repeat_interleave(output[:,aux_idxs], 5, dim=1).reshape(len(aux_idxs),5,256).reshape(len(aux_idxs),5*256)
                     aux_pos = self.embs_id(aux_pos).unsqueeze(0)
                     track_queries['pos_emb'] = torch.cat((track_queries['pos_emb'],aux_pos),dim=1)
            
            n_max = torch.max(track_queries['ids'])
            if len(aux_idxs) > 0:
               track_queries['ids'] = torch.cat((track_queries['ids'],torch.arange(n_max+1,n_max+len(aux_idxs)+1)))
               assert track_queries['ids'].shape[0] == track_queries['boxes'].shape[1]

      #print('track boxes:',track_queries['boxes'])
      if self.training:
         return losses_dict, track_queries
      else:
         return out, track_queries

   @torch.jit.unused
   def _set_aux_loss(self, outputs_class, outputs_coord):
      # this is a workaround to make torchscript happy, as torchscript
      # doesn't support dictionary with non-homogeneous values, such
      # as a dict having both a Tensor and a list.
      return [{'pred_logits': a, 'pred_boxes': b}
              for a, b in zip(outputs_class, outputs_coord)]
