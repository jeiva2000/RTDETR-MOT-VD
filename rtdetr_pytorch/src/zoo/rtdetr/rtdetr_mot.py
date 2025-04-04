"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register
import time

__all__ = ['RTDETR_MOT', ]


@register
class RTDETR_MOT(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'criterion']

    def __init__(self, backbone: nn.Module, encoder, decoder, criterion, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.criterion = criterion
        self.multi_scale = multi_scale
        
        #self.hidden_dim = 256
        #self.num_queries = 300
        #self.query_embed = nn.Embedding(self.num_queries,self.hidden_dim)
        """
        self.position = nn.Embedding(self.num_queries,4)
        self.id_embed = nn.Embedding(self.num_queries,301)
        nn.init.uniform_(self.position.weight.data, 0, 1)
        """

    def init_tracks(self,):
        #self.queries = self.query_embed.weight.to("cuda")
        #self.track_ref_points = self.position.weight.to("cuda")
        #self.track_queries = {'track_queries':None,'hs':None,'h_points':None,
        #                      'pred_boxes':torch.zeros(1,self.num_queries,4).to('cuda'),'pred_logits':torch.zeros(1,self.num_queries,2).to('cuda'),
        #                      'query_pos':self.position.weight,"id_embed":self.id_embed.weight}
        self.track_queries = None
        self.losses = {}

    def forward(self,x,targets=None):
        self.init_tracks()
        if self.training:
           cont=0
           #print("len images:",len(x))
           for fx,target in zip(x,targets):
               #fx = fx.unsqueeze(0)
               target = [target]
               #print("target:",target)
               fx, track_queries = self.forward_single(fx,target,self.track_queries)
               #print("sale forward:",track_queries["track_queries"])
               losses_dict, self.track_queries = self.criterion.match(fx, track_queries, target)
               #print('sale track_queries:',self.track_queries['track_queries'])
               for k,v in losses_dict.items():
                   if k not in self.losses:
                      self.losses[k] = 0
                   self.losses[k]+=v
               cont+=1
               #print("cont:",cont)
           for k,v in losses_dict.items(): #probando por el calculo del error en cada imagen
               self.losses[k]/=len(x)
           return self.losses
        else:
          outputs = []
          outputs_2 = []
          for fx in x:
              #fx = fx.unsqueeze(0)
              start_time = time.time()
              #print('fx:',fx.shape)
              fx, fx_2 = self.forward_single(fx,None,self.track_queries)
              #print("--- %s seconds ---" % (time.time() - start_time))
              outputs.append(fx)
              outputs_2.append(fx_2)
          #print('outputs_2:',outputs_2)
          if outputs_2[-1] is not None and outputs_2[-1]['track_queries'] is not None:
             return outputs_2
          else:
             return outputs

    def forward_single(self, x, targets=None,track_queries=None,limit=2,ind_track=True):
        if self.multi_scale and self.training:
           sz = np.random.choice(self.multi_scale)
           #print("x shape:",x.shape)
           x = F.interpolate(x, size=[sz, sz])
        x = self.backbone(x)
        x = self.encoder(x)
        x, track_queries = self.decoder(x,targets,track_queries)
        return x, track_queries
   
    @torch.no_grad()
    def forward_inference_2(self, x, track_queries, ind_track=True, n_frame=None, h_limit=None):
        fx, track_queries = self.forward_single(x,None,track_queries,limit=10,ind_track=ind_track)
        return fx, track_queries

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
