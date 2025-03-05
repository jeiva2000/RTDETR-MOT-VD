from .rtdetr_criterion import SetCriterion
from src.core import register
import torch
from src.misc.dist import get_world_size, is_dist_available_and_initialized
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from pytorch_metric_learning import losses as l_c
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou


def nt_bxent_loss(x, pos_indices, temperature):
    assert len(x.size()) == 2

    # Add indexes of the principal diagonal elements to pos_indices
    if pos_indices.size(0) == 0:
      pos_indices = torch.arange(x.size(0),device=x.device).reshape(x.size(0), 1).expand(-1, 2)
    else:
      pos_indices = torch.cat([
          pos_indices,
          torch.arange(x.size(0)).reshape(x.size(0), 1).expand(-1, 2),
          ], dim=0)
    
    # Ground truth labels
    target = torch.zeros(x.size(0), x.size(0),device=x.device)
    target[pos_indices[:,0], pos_indices[:,1]] = 1.0

    #print('target:',target)

    # Cosine similarity
    xcs = F.cosine_similarity(x[None,:,:], x[:,None,:], dim=-1)
    # Set logit of diagonal element to "inf" signifying complete
    # correlation. sigmoid(inf) = 1.0 so this will work out nicely
    # when computing the Binary cross-entropy Loss.
    xcs[torch.eye(x.size(0)).bool()] = float("inf")

    # Standard binary cross-entropy loss. We use binary_cross_entropy() here and not
    # binary_cross_entropy_with_logits() because of
    # https://github.com/pytorch/pytorch/issues/102894
    # The method *_with_logits() uses the log-sum-exp-trick, which causes inf and -inf values
    # to result in a NaN result.
    loss = F.binary_cross_entropy((xcs / temperature).sigmoid(), target, reduction="none")
    target_pos = target.bool()
    target_neg = ~target_pos
    loss_pos = torch.zeros(x.size(0), x.size(0),device=x.device).masked_scatter(target_pos, loss[target_pos])
    loss_neg = torch.zeros(x.size(0), x.size(0),device=x.device).masked_scatter(target_neg, loss[target_neg])
    loss_pos = loss_pos.sum(dim=1)
    loss_neg = loss_neg.sum(dim=1)
    num_pos = target.sum(dim=1)
    num_neg = x.size(0) - num_pos
    loss = loss_pos / num_pos
    if num_neg.any():
      loss+=(loss_neg / num_neg)
    return loss.mean()

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

      def match(self, outputs, track_queries, targets, det_boxes=None, th_filter=0.5, mode=''):
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
          if 'cons_track_aux' in track_queries:
            cons_track = track_queries['cons_track_aux']
          else:
            cons_track = None
          if 'cons_det_aux' in track_queries:
            cons_det = track_queries['cons_det_aux']
          else:
            cons_det = None

          if cons_track is not None and cons_det is not None:
            aux_constrack = torch.cat((cons_track,cons_det),0)
            ious_re = F.normalize(aux_constrack) @ F.normalize(aux_constrack).t()
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
            idxs = torch.tensor(idxs,device=aux_boxes.device)
            
            """
            #reid ini -- revisaar
            if 'projs_reid' in track_queries and track_queries['projs_reid'] is not None:
              aux_ious_gt = generalized_box_iou(box_cxcywh_to_xyxy(aux_boxes),box_cxcywh_to_xyxy(targets[0]['boxes']))
              v, ids = torch.max(aux_ious_gt,1)
              ids = ids[v>0.5]
              gts = targets[0]['ids'][ids]
              ids_gt_dict = {gt.item():idxs[j] for j,gt in enumerate(gts)}
              #print('ids_gt_dict:',ids_gt_dict)
              gt_ids_hist = track_queries['gt_ids_hist'].clone()
              projs_reid = track_queries['projs_reid'].clone()
              ids_gt = torch.tensor(list(ids_gt_dict.keys()),device=aux_boxes.device)
              aux_mask = torch.logical_and(gt_ids_hist!=-1,torch.isin(gt_ids_hist,ids_gt))
              gt_ids_hist = gt_ids_hist[aux_mask]
              projs_reid = projs_reid[aux_mask]
              projs_id = [ids_gt_dict[gt.item()] for gt in gt_ids_hist]
              projs_id = torch.tensor(projs_id,device=projs_reid.device)
              aux_constrack = torch.cat((aux_constrack,projs_reid),0)
              idxs = torch.cat((idxs,projs_id),0)
            #reid fin
            """

            num_idxs = len(torch.unique(idxs))
            if num_idxs == len(idxs) or num_idxs == 1:
              aux_constrack = torch.repeat_interleave(aux_constrack,2,dim=0)
              idxs = torch.repeat_interleave(idxs,2)
              aux_ious = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['boxes'].squeeze(0)),box_cxcywh_to_xyxy(outputs_without_aux['pred_boxes'].squeeze(0)))
              v, ids = torch.min(aux_ious,1)
              ids = ids[v<0.5]
              if len(ids)>0:
                ids = ids[0]
                idxs = torch.cat((idxs,torch.tensor([torch.max(idxs)+1],device=idxs.device)))
                aux_constrack = torch.cat((aux_constrack,track_queries['cons_det_aux_2'][:,ids]),0)
              

            l_dict = {'distance_track':self.cons_loss(aux_constrack,idxs)}

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

          """
          #reid
          if 'cons_det' in track_queries:
            cons_det = track_queries['cons_det']
          else:
            cons_det = None

          if 'projs_reid' in track_queries and track_queries['projs_reid'] is not None and cons_det is not None and det_boxes is not None:
            print('det_boxes shape:',det_boxes.shape)
            print('targets shape:',targets[0]['boxes'].shape)
            aux_ious = generalized_box_iou(box_cxcywh_to_xyxy(det_boxes),box_cxcywh_to_xyxy(targets[0]['boxes']))
            v, ids = torch.max(aux_ious,1)
            det_ids = targets[0]['ids'][ids[v>0.5]]
            new_ids = torch.cat((track_queries['gt_ids_hist'],det_ids))
            aux_constrack = torch.cat((track_queries['projs_reid'],cons_det[v>0.5]))
            print('aux_constrack shape:',aux_constrack.shape)
            print('new_ids shape:',new_ids.shape)
            print('new_ids:',new_ids)
            ious_re = F.normalize(aux_constrack) @ F.normalize(aux_constrack).t()
            print('ious_re 2:',ious_re)
            num_idxs = len(torch.unique(new_ids))
            if num_idxs == len(new_ids) or num_idxs == 1:
              aux_constrack = torch.repeat_interleave(aux_constrack,2,dim=0)
              new_ids = torch.repeat_interleave(new_ids,2)
            l_dict = {'distance_track_reid':self.cons_loss(aux_constrack,new_ids)}
            print('distance_track_reid:',l_dict['distance_track_reid'])
            losses_dict.update(l_dict)
          """
          return losses_dict, track_queries
