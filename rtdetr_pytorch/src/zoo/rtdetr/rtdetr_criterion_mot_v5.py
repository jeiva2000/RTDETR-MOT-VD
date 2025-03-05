from .rtdetr_criterion import SetCriterion
from src.core import register
import torch
from src.misc.dist import get_world_size, is_dist_available_and_initialized
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from pytorch_metric_learning import losses as l_c
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
import torchvision 

@register
class CriterionMOT_v2(SetCriterion):
      def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e4, num_classes=80):
          super().__init__(matcher, weight_dict, losses,alpha, gamma, eos_coef, num_classes)
          self.losses = losses
          self.cons_loss = l_c.SupConLoss(temperature=0.1)

      def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

      def match(self, outputs, track_queries, targets, th_filter=0.5, mode=''):
          losses_dict = {}

          num_boxes = sum(len(t["labels"]) for t in targets)
          num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
          if is_dist_available_and_initialized():
             torch.distributed.all_reduce(num_boxes)
          num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

          outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
          indices = self.matcher(outputs_without_aux, targets)
          indices_pred = indices[0][0].clone()
          indices_det = indices[0][1].clone()
          for loss in self.losses:
              l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
              l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
              losses_dict.update(l_dict)

          if 'aux_outputs' in outputs:
             for i, aux_outputs in enumerate(outputs['aux_outputs']):
               indices = self.matcher(aux_outputs, targets)
               for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses_dict.update(l_dict)

          if 'pred_logits_2_b' in track_queries:
              """
              if track_queries['mask_pred'].shape[0]==0:
                track_queries['mask_pred'] = None
                return losses_dict, track_queries
              ious = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['pred_boxes_2_b'].squeeze(0)),box_cxcywh_to_xyxy(outputs_without_aux['pred_boxes'][track_queries['mask_pred']]))
              ious = (ious > 0.5).float()
             
              corr = track_queries['corr'].squeeze(0)
            
              corr = corr[:,track_queries['mask_pred'].squeeze(0)]
              
              if ious.shape[1] > 0 and corr.shape[1] > 0 and torch.sum(ious) > 0:
                print('ious shape:',ious.shape)
                print('corr shape:',torch.sigmoid(corr).shape)
                print('ious:',ious)
                print('corr:',torch.sigmoid(corr))
                l_dict = {'corr_track':F.binary_cross_entropy_with_logits(corr.flatten(), ious.flatten())}
                print('loss:',l_dict['corr_track'])
                losses_dict.update(l_dict)
              """
              #track losses
              if track_queries['pred_boxes_2_b'].shape[1] > 0:
                aux_track_queries = {'pred_logits':track_queries['pred_logits_2_b'],'pred_boxes':track_queries['pred_boxes_2_b']}
                indices = self.matcher(aux_track_queries, targets)
                for loss in self.losses:
                  l_dict = self.get_loss(loss, aux_track_queries, targets, indices, num_boxes)
                  l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                  l_dict = {k + f'_track': v for k, v in l_dict.items()}
                  losses_dict.update(l_dict)
                #print('pasaa')

                #reconstruction losses
                #print('re logits:',track_queries['re_logits'].shape)
                #print('re boxes:',track_queries['re_boxes'].shape)
              if 're_boxes' in track_queries and track_queries['re_boxes'] is not None  and track_queries['re_boxes'].shape[1]>0:
                  aux_track_queries = {'pred_logits':track_queries['re_logits'],'pred_boxes':track_queries['re_boxes']}
                  #print('re_boxes:',track_queries['re_boxes'])
                  indices = self.matcher(aux_track_queries, targets)
                  for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_track_queries, targets, indices, num_boxes)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_track_re': v for k, v in l_dict.items()}
                    losses_dict.update(l_dict)

                  #print('pred boxes mask shape:',outputs_without_aux['pred_boxes'][track_queries['mask_pred']].shape)
                  #ious = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['pred_boxes_2_b'].squeeze(0)),box_cxcywh_to_xyxy(outputs_without_aux['pred_boxes'].squeeze(0)))
                  
                  #ious = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['pred_boxes_2_b'].squeeze(0)),box_cxcywh_to_xyxy(outputs_without_aux['pred_boxes'][track_queries['mask_pred']]))
                  ious = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['pred_boxes_2_b'].squeeze(0)),box_cxcywh_to_xyxy(outputs_without_aux['pred_boxes'][track_queries['mask_pred']]))
                  ious = (ious > 0.5).float()
                  #aux_diag = torch.eye(track_queries['pred_boxes_2_b'].shape[1],track_queries['pred_boxes_2_b'].shape[1]).to('cuda')
                  #ious = torch.cat((ious,aux_diag),dim=1)#add diagonal track
                  #track_queries['mask'].requires_grad = True
                  print('verify grad:',track_queries['mask'].requires_grad)
                  print('ious:',ious)
                  print('mask:',track_queries['mask'])
                  l_dict = {'res_track':F.binary_cross_entropy_with_logits(track_queries['mask'].flatten(),ious.flatten(),pos_weight=torch.tensor([10]).to('cuda'))} #probandoo
                  #l_dict = {'res_track':F.binary_cross_entropy(track_queries['mask'].flatten(),ious.flatten())} #probandoo
                  losses_dict.update(l_dict)
                  print('loss:',l_dict['res_track'])             

          #track_queries['mask_pred'] = None

          return losses_dict, track_queries
