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
          #print('targets:',targets)
          num_boxes = sum(len(t["labels"]) for t in targets)
          num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
          if is_dist_available_and_initialized():
             torch.distributed.all_reduce(num_boxes)
          num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
          #"""detection
          outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
          #print('outputs_without_aux:',outputs_without_aux)
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
          
          #Track
          """
          for i, aux_boxes in enumerate(outputs['aux_out_boxes_0']):
            #cost_bbox = torch.cdist(aux_boxes, targets[0]['boxes'], p=1).squeeze(0)
            cost_bbox = generalized_box_iou(box_cxcywh_to_xyxy(aux_boxes.squeeze(0)), box_cxcywh_to_xyxy(targets[0]['boxes']))
            indices = torch.nonzero(cost_bbox > 0.5)
            if indices.shape[0]==0:
              indices = [[i,0] for i, v in enumerate(targets[0]['boxes'])]
            indices = [(torch.nonzero(cost_bbox > 0.5)[:,0],torch.nonzero(cost_bbox > 0.5)[:,1])]
            aux_outputs = {'pred_boxes':outputs['aux_out_boxes_0'][i],'pred_logits':outputs['aux_out_logits_0'][i]}
            for loss in self.losses:
              l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
              l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
              l_dict = {k + f'_aux_match_0_{i}': v for k, v in l_dict.items()}
              losses_dict.update(l_dict)
          
          if 'aux_boxes' in track_queries and track_queries['aux_boxes'] is not None:
            for i, aux_boxes in enumerate(outputs['aux_out_boxes']):
              aux_boxes_0 = track_queries['aux_boxes']
              aux_logits_0 = track_queries['aux_logits']
              aux_boxes_1 = outputs['aux_out_boxes'][:,aux_boxes.shape[1]:]
              aux_logits_1 = outputs['aux_out_logits'][:,aux_boxes.shape[1]:]
              print('aux_boxes shape:',aux_boxes.shape)
              print('aux_boxes_1 shape:',aux_boxes_1.shape)
              cost_bbox = generalized_box_iou(box_cxcywh_to_xyxy(aux_boxes), box_cxcywh_to_xyxy(aux_boxes_1))
              #cost_bbox = torch.cdist(aux_boxes, aux_boxes_1, p=1).squeeze(0)
              indices = torch.nonzero(cost_bbox > 0.5)
              if indices.shape[0]==0:
                indices = [[i,0] for i, v in enumerate(targets[0]['boxes'])]
              indices = [(torch.nonzero(cost_bbox > 0.5)[:,0],torch.nonzero(cost_bbox > 0.5)[:,1])]
              aux_outputs = {'pred_boxes':aux_boxes_1[i],'pred_logits':aux_logits_1[i]}
              aux_targets = [{'labels':aux_logits_0,'boxes':aux_boxes_0}]
              for loss in self.losses:
                l_dict = self.get_loss(loss, aux_outputs, aux_targets, indices, num_boxes)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + f'_aux_match_1_{i}': v for k, v in l_dict.items()}
                losses_dict.update(l_dict)

          #minimize attention scores
          if 'attention_scores' in track_queries:
            print('attention scores shape:',track_queries['attention_scores'].shape)
            print('boxes shape:',track_queries['boxes'].shape)
            cost_bbox = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['boxes'].squeeze(0)), box_cxcywh_to_xyxy(track_queries['boxes'].squeeze(0)))
            corr_loss = torch.nn.functional.mse_loss(track_queries['attention_scores'],cost_bbox)
            l_dict = {f'_corr_loss': corr_loss}
            losses_dict.update(l_dict)
          """
          
          #minimize attention scores
          if 'attention_scores' in outputs_without_aux:
            if 'boxes' in track_queries:
              aux_boxes = torch.cat((outputs_without_aux['pred_boxes'],track_queries['boxes']),dim=1)
            else:
              aux_boxes = outputs_without_aux['pred_boxes'].clone()
            cost_bbox = generalized_box_iou(box_cxcywh_to_xyxy(aux_boxes.squeeze(0)), box_cxcywh_to_xyxy(aux_boxes.squeeze(0)))
            #corr_loss = torch.nn.functional.mse_loss(outputs_without_aux['attention_scores'],cost_bbox)
            cost_bbox[cost_bbox < 0] = 0
            cost_bbox = cost_bbox.detach()
            cost_bbox.requires_grad=False
            """
            with torch.autograd.detect_anomaly():
              print('attention scores:',outputs_without_aux['attention_scores'])
              print('grad input:',outputs_without_aux['attention_scores'].requires_grad)
              print('targets:',cost_bbox)
              print('grad target:',cost_bbox.requires_grad)
              corr_loss = torch.nn.functional.binary_cross_entropy(outputs_without_aux['attention_scores'],cost_bbox)
            """
            #corr_loss = torch.nn.functional.binary_cross_entropy(outputs_without_aux['attention_scores'],cost_bbox)
            #aux_attn = outputs_without_aux['attention_scores'].clone()
            #aux_attn[aux_attn<0] = 0
            #outputs_without_aux['attention_scores'] = aux_attn
            #print('grad:',outputs_without_aux['attention_scores'].requires_grad)
            corr_loss = torch.nn.functional.binary_cross_entropy(outputs_without_aux['attention_scores'],cost_bbox)
            #corr_loss = torch.nn.functional.mse_loss(outputs_without_aux['attention_scores'],cost_bbox)
            l_dict = {f'corr_loss': corr_loss}
            #print('corr loss:',l_dict)
            losses_dict.update(l_dict)

          """
          #minimize attention scores with contrastive learning
          if 'attention_scores' in outputs_without_aux:
            pair_dict = {}
            if 'boxes' in track_queries:
              aux_boxes = torch.cat((outputs_without_aux['pred_boxes'],track_queries['boxes']),dim=1)
            else:
              aux_boxes = outputs_without_aux['pred_boxes'].clone()
            cost_bbox = generalized_box_iou(box_cxcywh_to_xyxy(aux_boxes.squeeze(0)), box_cxcywh_to_xyxy(aux_boxes.squeeze(0)))
            for j,aux in enumerate(cost_bbox):
              values = [v for k,v in pair_dict.items()]
              if j not in values:
                pair_dict[j] = torch.nonzero(aux>0.5)[:,1].detach().cpu().numpy()
            l_dict = {'corr_loss':self.cons_loss(track_queries['embeds'],idxs)}
          """
          return losses_dict, track_queries
