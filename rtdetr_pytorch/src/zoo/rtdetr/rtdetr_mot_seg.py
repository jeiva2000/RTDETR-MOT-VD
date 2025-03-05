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
    
    def init_tracks(self):
        self.track_queries = {}
        self.losses = {}

    def forward(self,fx,track_queries,targets=None,epoch=None,return_two_outputs=False,last=False):
      target = [targets]
      fx_orig = fx.clone()
      if epoch < 20:
        fx, track_queries = self.forward_single(fx,target,track_queries,ind_track=False)
      else:
        fx, track_queries = self.forward_single(fx,target,track_queries)
      if self.training:
        losses_dict, track_queries = self.criterion.match(fx, track_queries, target, fx_orig)
        if last:
          losses_dict = self.criterion.calc_seg_loss(fx_orig,losses_dict, track_queries)
        return losses_dict, track_queries
      else:
        if return_two_outputs:
          return fx, track_queries
        else:
          return track_queries
    """
    def forward(self,x,targets=None,epoch=None,return_two_outputs=False):
        self.init_tracks()
        if self.training:
           cont=0
           for fx,target in zip(x,targets):
               target = [target]

               if epoch < 2:
                #print('x shape:',fx.shape)
                start_time = time.time()
                fx, track_queries = self.forward_single(fx,target,self.track_queries,ind_track=False)
                #print('cont:',cont)
                #print("--- %s seconds ---" % (time.time() - start_time))
               else:
                fx, track_queries = self.forward_single(fx,target,self.track_queries)
               
               #fx, track_queries = self.forward_single(fx,target,self.track_queries)
               losses_dict, self.track_queries = self.criterion.match(fx, track_queries, target)
               for k,v in losses_dict.items():
                   if k not in self.losses:
                      self.losses[k] = 0
                   self.losses[k]+=v
               cont+=1
           start_time = time.time()
           for k,v in losses_dict.items():
               self.losses[k]/=len(x)
           #print("--- %s seconds ---" % (time.time() - start_time))
           return self.losses
        else:
          outputs = []
          outputs_2 = []
          for fx in x:
              start_time = time.time()
              
              if epoch < 2:
                fx, self.track_queries = self.forward_single(fx,None,self.track_queries,ind_track=False)
              else:
                fx, self.track_queries = self.forward_single(fx,None,self.track_queries)
              
              #fx, self.track_queries = self.forward_single(fx,None,self.track_queries)
              if return_two_outputs:
                outputs_2.append(fx)
              #print('pred_boxes antes de agregar:',self.track_queries['pred_boxes_2'])
              outputs.append(self.track_queries.copy()) #cambio para ir probando, volver a colocar
          if return_two_outputs:
            return outputs_2 ,outputs
          else:
            return outputs
    """
    def forward_inference(self,x):
        fx, self.track_queries = self.forward_single(x,None,self.track_queries)
        return fx, self.track_queries

    @torch.no_grad()
    def forward_inference_2(self, x, track_queries):
       fx, track_queries = self.forward_single(x,None,track_queries)
       #fx, track_queries = self.forward_single(x,None,track_queries,ind_track=False)
       return fx, track_queries

    def forward_single(self, x, targets=None, track_queries=None, ind_track=True):
        if self.multi_scale and self.training:
           sz = np.random.choice(self.multi_scale)
           x = F.interpolate(x, size=[sz, sz])
        x = self.backbone(x)
        x_back = x.copy()
        x = self.encoder(x)
        x, track_queries = self.decoder(x,targets,track_queries,ind_track,x_back)
        return x, track_queries

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
