"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register
import time

__all__ = ['RTDETR_MOT_v2', ]


@register
class RTDETR_MOT_v2(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'criterion']

    def __init__(self, backbone: nn.Module, encoder, decoder, criterion, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.criterion = criterion
        self.multi_scale = multi_scale
        self.empty_embed = nn.Embedding(1,256)
        self.query_pos_embed = nn.Embedding(1,256)
    
    def init_tracks(self):
        self.track_queries = {'embeds':None,"ids":None}
        self.losses = {}
    """
    def get_track_queries(self):
        return self.track_queires
    """
    def forward(self,x,targets=None,epoch=None):
        self.init_tracks()
        if self.training:
           cont=0
           #print("len images:",len(x))
           #print('inicia batch')
           for fx,target in zip(x,targets):
               #fx = fx.unsqueeze(0)
               target = [target]
               #print("target:",target)
               """
               if epoch < 1:
                  fx, track_queries = self.forward_single(fx,target,self.track_queries,ind_track=False)
               else:
                  fx, track_queries = self.forward_single(fx,target,self.track_queries)
               """
               fx, track_queries = self.forward_single(fx,target,self.track_queries)
               #print("sale forward:",track_queries["track_queries"])
               losses_dict, self.track_queries = self.criterion.match(fx, track_queries, target)
               #print('sale track_queries:',self.track_queries['track_queries'])
               for k,v in losses_dict.items():
                   if k not in self.losses:
                      self.losses[k] = 0
                   self.losses[k]+=v
                   #if "loss_giou_track" == k and v > 1.0:
                   #   print("targets:",target)
                   #   print("track_queries:",self.track_queries["pred_boxes"])
               cont+=1
               #print("cont:",cont)
           for k,v in losses_dict.items(): #probando por el calculo del error en cada imagen
               self.losses[k]/=len(x)
           #print('finaliza batch')
           return self.losses
        else:
          outputs = []
          #outputs_2 = []
          print("empieza secuencia")
          for fx in x:
              #fx = fx.unsqueeze(0)
              start_time = time.time()
              #print('fx:',fx.shape)
              fx, self.track_queries = self.forward_single(fx,None,self.track_queries)
              #print("--- %s seconds ---" % (time.time() - start_time))
              #outputs.append(fx)
              #outputs_2.append(self.track_queries)
              """
              if epoch < 1:
                 fx, self.track_queries = self.forward_single(fx,None,self.track_queries,ind_track=False)
                 outputs.append(fx)
              else:
                 fx, self.track_queries = self.forward_single(fx,None,self.track_queries)
                 outputs.append(self.track_queries)
              """
              outputs.append(self.track_queries)
          """
          #print('outputs_2:',outputs_2)
          if outputs_2[-1] is not None and outputs_2[-1]['embeds'] is not None:
             print("devuelve salida track")
             return outputs_2
          else:
             print("devuelve salida normal")
             return outputs
          """
          return outputs

    def forward_inference(self,x):
        fx, self.track_queries = self.forward_single(x,None,self.track_queries)
        return fx, self.track_queries

    def forward_inference_2(self, x, track_queries):
       fx, track_queries = self.forward_single(x,None,track_queries)
       return fx, track_queries

    def forward_single(self, x, targets=None, track_queries=None, ind_track=True):
        if self.multi_scale and self.training:
           sz = np.random.choice(self.multi_scale)
           #print("x shape:",x.shape)
           x = F.interpolate(x, size=[sz, sz])
        x = self.backbone(x)
        x = self.encoder(x)
        x, track_queries = self.decoder(x,targets,track_queries, ind_track)
        return x, track_queries

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
