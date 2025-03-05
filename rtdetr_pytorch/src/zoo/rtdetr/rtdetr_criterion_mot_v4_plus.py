from .rtdetr_criterion import SetCriterion
from src.core import register
import torch
from src.misc.dist import get_world_size, is_dist_available_and_initialized
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from pytorch_metric_learning import losses as l_c
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

"""
def fusion_boxes(matrix):
  for row in matrix:
"""

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean().sum() / num_boxes

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
                    
          if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses_dict.update(l_dict)

          #Calculate distance loss
          if 'cons_track' in track_queries and 'cons_det' in track_queries and track_queries['cons_det'] is not None:
            aux_constrack = torch.cat((track_queries['cons_track'],track_queries['cons_det']),0)
            aux_boxes = torch.cat((track_queries['boxes'].squeeze(0),outputs_without_aux['pred_boxes'][track_queries['mask_pred']]))
            aux_ious = generalized_box_iou(box_cxcywh_to_xyxy(aux_boxes),box_cxcywh_to_xyxy(aux_boxes))
            aux_ious = torch.nonzero(aux_ious > 0.5)
            aux_dict = {}
            aux_list = []

            for v in aux_ious:
              k,v = v[0].item(), v[1].item()
              if k not in aux_dict:
                aux_dict[k] = []
              if v not in aux_list:
                aux_dict[k].append(v)
                aux_list.append(v)
            idxs = []
            count = 0
            for i in range(len(aux_constrack)):
              idx = [k for k,v in aux_dict.items() if i in v]
              if len(idx) == 0:
                count+=1
                idx = count
              else:
                idx = idx[0]
              idxs.append(idx)
            idxs = torch.tensor(idxs)
            l_dict = {'distance_track':self.cons_loss(aux_constrack,idxs)}
            #print('aux_constrack shape:',aux_constrack.shape)
            #print('idxs distance:',idxs)
            #print('loss distance:',l_dict['distance_track'])
            losses_dict.update(l_dict)

            #track loss
            if track_queries['boxes'].shape[1] > 0:
              aux_track_queries = {'pred_logits':track_queries['logits'],'pred_boxes':track_queries['boxes']}
              indices = self.matcher(aux_track_queries, targets)
              for loss in self.losses:
                l_dict = self.get_loss(loss, aux_track_queries, targets, indices, num_boxes)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + f'_track': v for k, v in l_dict.items()}
                losses_dict.update(l_dict)

            
            if 'ids_hist' in track_queries and track_queries['ids_hist'].shape[0] > 0 and track_queries['cons_det'] is not None:
              aux_constrack = torch.cat((track_queries['projs_reid'],track_queries['cons_det']),0)
              aux_ious = generalized_box_iou(box_cxcywh_to_xyxy(outputs_without_aux['pred_boxes'][track_queries['mask_pred']]),box_cxcywh_to_xyxy(targets[0]['boxes']))
              v, idxs = torch.max(aux_ious,1)
              idxs_det = idxs[v>0.5]
              idxs = torch.cat((track_queries['gt_ids_hist'],targets[0]['ids'][idxs_det]))
              l_dict = {'distance_track_reid':self.cons_loss(aux_constrack,idxs)}
              losses_dict.update(l_dict)
            
            
          return losses_dict, track_queries
