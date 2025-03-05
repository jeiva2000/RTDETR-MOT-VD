from .rtdetr_criterion import SetCriterion
from src.core import register
import torch
from src.misc.dist import get_world_size, is_dist_available_and_initialized
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from pytorch_metric_learning import losses as l_c
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        #return l_c.NTXentLoss(self.temperature)(logits, torch.squeeze(labels))
        return l_c.SupConLoss(self.temperature)(logits, torch.squeeze(labels))
        #return l_c.SupConLoss(feature_vectors, torch.squeeze(labels))

@register
class CriterionMOT_v2(SetCriterion):
      def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e4, num_classes=80):
          super().__init__(matcher, weight_dict, losses,alpha, gamma, eos_coef, num_classes)
          self.losses = losses
          self.cons_loss = l_c.SupConLoss(temperature=0.1)
          #self.cons_loss = SupervisedContrastiveLoss()

      def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

      def match(self, outputs, targets, track_queries, det_boxes=None, th_filter=0.5, mode=''):
          losses_dict = {}
          device = outputs['embeds'].device

          num_boxes = sum(len(t["labels"]) for t in targets)
          num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
          if is_dist_available_and_initialized():
             torch.distributed.all_reduce(num_boxes)
          num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

          outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
          indices = self.matcher(outputs_without_aux, targets)
          indices_pred = indices[0][0].clone()
          indices_gt = indices[0][1].clone()
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

          embeds = outputs_without_aux['embeds'][:,indices_pred].clone().squeeze(0)
          idxs = targets[0]['ids'][indices_gt]
          idxs = idxs.to(device)

          if 'embeds' in track_queries:
            #print('track shape:',track_queries['cons_embs'].shape)
            #print('track shape:',track_queries['embeds'].shape)
            #print('idx shape:',track_queries['ids_gt'].shape)
            embeds = torch.cat((embeds,track_queries['cons_embs'].squeeze(0)))
            idxs = torch.cat((idxs,track_queries['ids_gt']))

          #Agregando mas predicciones
          scores = F.sigmoid(outputs_without_aux['pred_logits'])
          mask = torch.max(scores,2)[0] > th_filter
          mask = mask.squeeze(0)

          if outputs_without_aux['pred_boxes'][:,mask].shape[1] > 0 and targets[0]['boxes'].shape[0]>0:
            aux_ious = generalized_box_iou(box_cxcywh_to_xyxy(outputs_without_aux['pred_boxes'][:,mask].squeeze(0)),box_cxcywh_to_xyxy(targets[0]['boxes']))
            v, aux_idxs = torch.max(aux_ious,1)
            if (v>0.5).sum() > 0:
              embeds = torch.cat((embeds,outputs_without_aux['embeds'][:,mask].squeeze(0)[v>0.5]))
              idxs = torch.cat((idxs,targets[0]['ids'][aux_idxs[v>0.5]]))

          #aumentando
          embeds = torch.repeat_interleave(embeds,2,dim=0)
          idxs = torch.repeat_interleave(idxs,2)

          l_dict = {'distance_track':self.cons_loss(embeds,idxs)}
          #print('dist:',dist>0.5)
          #print(torch.eq(idxs.reshape(idxs.shape[0],1),idxs))
          losses_dict.update(l_dict)

          preds = F.normalize(embeds) @ F.normalize(embeds).t()
          preds = preds > 0.5
          preds = preds.float()
          preds.requires_grad = True
          gts_t = torch.eq(idxs.reshape(idxs.shape[0],1),idxs).float()
          losses_dict.update({'distance_1_track': F.binary_cross_entropy(preds,gts_t)*5})
          #print(l_dict)
          return losses_dict, track_queries
