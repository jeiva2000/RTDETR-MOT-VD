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
          
          if 'pred_logits_2_b' in track_queries and track_queries['pred_logits_2_b'].shape[1] > 0:
            th = 0.5
            with torch.no_grad():
              scores = F.sigmoid(outputs_without_aux['pred_logits'])
              mask = torch.max(scores,2)[0] > th
            if mask.sum() > 0:
              ious_t = generalized_box_iou(box_cxcywh_to_xyxy(targets[0]['boxes']),box_cxcywh_to_xyxy(track_queries['pred_boxes_2_b'].squeeze(0)))
              ious_t, idx_t = torch.max(ious_t,1)
              ious_t = ious_t > 0.5
              idx_gt = torch.nonzero(ious_t.flatten())
              idx_t = idx_t[idx_gt]

              valid_ids = targets[0]['ids'][idx_gt]
              dict_valid_ids = {targets[0]['ids'][idx].item():idx.item() for idx in idx_gt}
              valid_pred = track_queries['ids_b'][idx_t]
              dict_valid_ids_det = {track_queries['ids_b'][idx].item():idx.item() for idx in idx_t}

              valid_gt = {v.item(): valid_pred[i].item() for i,v in enumerate(valid_ids)}

              if 'ids_gt' not in track_queries:
                track_queries['ids_gt'] = {t.item():None for t in targets[0]['ids']}

              track_queries['ids_gt'] = {k: (valid_gt[k] if v is None and k in valid_gt else v) for k,v in track_queries['ids_gt'].items() }

              aux_track_queries = {'pred_logits':track_queries['pred_logits_2_b'],'pred_boxes':track_queries['pred_boxes_2_b']}

              if len(dict_valid_ids_det) > 0 and len(dict_valid_ids_det) > 0:
                indices  = [[torch.tensor([dict_valid_ids_det[v]]),torch.tensor([dict_valid_ids[k]])] for k,v in track_queries['ids_gt'].items() if v is not None and k in dict_valid_ids and v in dict_valid_ids_det]
              else:
                indices = []

              if len(indices) > 0:
                indices = [(torch.stack([row[0].squeeze(0) for row in indices]),torch.stack([row[1].squeeze(0) for row in indices]))]
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_track_queries, targets, indices, num_boxes)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_track': v for k, v in l_dict.items()}
                    losses_dict.update(l_dict)
            #print('cons_track shape antes:',track_queries['cons_track'].shape)
            #print('cons_det shape antes:',track_queries['cons_det'].shape)
            aux_constrack = torch.cat((track_queries['cons_track'].clone(),track_queries['cons_det']),0)
            aux_boxes = torch.cat((track_queries['pred_boxes_2_b'].squeeze(0),outputs_without_aux['pred_boxes'][track_queries['mask_pred']]))
            aux_ious = generalized_box_iou(box_cxcywh_to_xyxy(aux_boxes),box_cxcywh_to_xyxy(aux_boxes))
            #print('aux_ious:',aux_ious)
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

            #print('aux_dict:',aux_dict)
            idxs = []
            count = 0
            for i in range(len(aux_constrack)):
              idx = [k for k,v in aux_dict.items() if i in v]
              if len(idx) == 0:
                count+=1
                idx = count
                #idx = -1
              else:
                idx = idx[0]
              idxs.append(idx)
            idxs = torch.tensor(idxs)
            #print('idxs:',idxs)
            l_dict = {'distance_track':self.cons_loss(aux_constrack,idxs)}
            #print('loss:',l_dict['distance_track'])
            losses_dict.update(l_dict)

            #""" Volver a colocar para seguir probando -- esta puede ser que quite
            #Probaaar reid
            if 'boxes_re' in track_queries and track_queries['boxes_re'] is not None:
              aux_track_queries = {'pred_logits':track_queries['scores_re'],'pred_boxes':track_queries['boxes_re']}
              indices = self.matcher(aux_track_queries, targets)
              for loss in self.losses:
                l_dict = self.get_loss(loss, aux_track_queries, targets, indices, num_boxes)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + f'_re': v for k, v in l_dict.items()}
                losses_dict.update(l_dict)
              track_queries['boxes_re'] = None
              track_queries['scores_re'] = None

            #Probando reducir el error de ids
            if 'dict_ids' in track_queries and track_queries['dict_ids'] is not None:
              idxs = [x for x in range(len(track_queries['dict_ids']))]
              for x in range(len(track_queries['hist_embed'])):
                ind = False
                for x2, (v) in enumerate(track_queries['dict_ids']):
                  if x in v:
                    idxs.append(x2)
                    ind = True
                    break
                if not ind:
                  idxs.append(-1)
              cont = sum([1 for aux in range(len(idxs)) if aux!=-1])
              for aux in range(len(idxs)):
                if idxs[aux] == -1:
                  idxs[aux] = cont 
                  cont+=1
              aux_constrack = torch.cat((track_queries['hist_embed_det'],track_queries['hist_embed']),0)
              idxs = torch.tensor(idxs)
              l_dict = {'distance_track_ids':self.cons_loss(aux_constrack,idxs)}
              losses_dict.update(l_dict)
              track_queries['dict_ids'] = None
              track_queries['hist_embed'] = None
              track_queries['hist_embed_det'] = None
            """
            ious = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['pred_boxes_2_b'].squeeze(0)),box_cxcywh_to_xyxy(outputs_without_aux['pred_boxes'][track_queries['mask_pred']]))
            ious = (ious > 0.5 ).float()
            print('ious:',ious)
            res = F.normalize(track_queries['cons_track']) @ F.normalize(track_queries['cons_det']).t()
            l_dict = {'res_track':F.binary_cross_entropy_with_logits(res.flatten(),ious.flatten(),pos_weight=torch.tensor([10]).to('cuda'))}
            losses_dict.update(l_dict)
            """

            """ comentariado para probar
            #print('validtracks shape:',valid_tracks.shape)
            #ious = generalized_box_iou(box_cxcywh_to_xyxy(valid_tracks),box_cxcywh_to_xyxy(outputs_without_aux['pred_boxes'][mask]))
            ious = generalized_box_iou(box_cxcywh_to_xyxy(valid_tracks),box_cxcywh_to_xyxy(outputs_without_aux['pred_boxes'][track_queries['mask_pred']]))
            ious = (ious > 0.5 ).float()
            #print('ious shape:',ious.shape)
            #print('cons_track shape:',track_queries['cons_track'].shape)
            res = F.normalize(track_queries['cons_track'][idx_t.flatten()] @ F.normalize(track_queries['cons_det']).t())
            #print('res shape:',res.shape)
            
            l_dict = {'res_track':F.binary_cross_entropy_with_logits(res.flatten(),ious.flatten())*3} #probandoo
            losses_dict.update(l_dict)

            #l_dict = {'res_track_focal':sigmoid_focal_loss(res.flatten(),ious.flatten(),num_boxes)} #probandoo
            #losses_dict.update(l_dict)
            """
          return losses_dict, track_queries
