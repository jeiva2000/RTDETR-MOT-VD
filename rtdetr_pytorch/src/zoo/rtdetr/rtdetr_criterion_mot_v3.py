from .rtdetr_criterion import SetCriterion
from src.core import register
import torch
from src.misc.dist import get_world_size, is_dist_available_and_initialized
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from pytorch_metric_learning import losses as l_c
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
#import torchvision

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        dist =  1 - F.normalize(output1) @ F.normalize(output2).t()
        pos = (1-label) * torch.pow(dist.flatten(), 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - dist.flatten(), min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive

@register
class CriterionMOT_v2(SetCriterion):
      def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e4, num_classes=80):
          super().__init__(matcher, weight_dict, losses,alpha, gamma, eos_coef, num_classes)
          self.losses = losses
          self.cons_loss = l_c.SupConLoss(temperature=0.1)
          self.cons_id_loss = nn.CrossEntropyLoss()

      def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

      def match(self, outputs, track_queries, targets, th_filter=0.5, mode=''):
          torch.autograd.set_detect_anomaly(True)
          losses_dict = {}

          num_boxes = sum(len(t["labels"]) for t in targets)
          num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
          if is_dist_available_and_initialized():
             torch.distributed.all_reduce(num_boxes)
          num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

          outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
          indices = self.matcher(outputs_without_aux, targets)
          #print('indices:',indices)
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
            th = 0.5
            with torch.no_grad():
              scores = F.sigmoid(outputs_without_aux['pred_logits'])
              mask = torch.max(scores,2)[0] > th
            if mask.sum() > 0:
              ious = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['pred_boxes_2_b'].squeeze(0)),box_cxcywh_to_xyxy(outputs_without_aux['pred_boxes'][mask]))
              #print('ious:',ious)
              ious_scores, ious_indexs = torch.max(ious,dim=-1)
              mask_iou = ious_scores > th
              aux_mask = ious > 0.5
              aux_mask = aux_mask.float()
              #aux_mask = torch.where(aux_mask > 0, 1.0, -1.0)
              aux_mask = torch.where(aux_mask > 0, 0.0, 1.0)
              #print('aux_mask:',aux_mask)
              #aux_embeds_2 = track_queries['embeds_2'].squeeze(0).clone()
              #aux_embeds_2 = track_queries['cons_track'].squeeze(0).clone()
              #l_dict = {'distance_track':F.cosine_embedding_loss(aux_embeds_2.repeat_interleave(track_queries['hs'][mask].shape[0],dim=0),track_queries['hs'][mask].repeat_interleave(aux_embeds_2.shape[0],dim=0),aux_mask.flatten())}
              #l_dict = {'distance_track':F.cosine_embedding_loss(aux_embeds_2.repeat_interleave(track_queries['cons_det'].shape[0],dim=0),track_queries['cons_det'].repeat_interleave(aux_embeds_2.shape[0],dim=0),aux_mask.flatten())}
              #print('ious:',ious)
              #print('aux_mask:',aux_mask)
              #print('aux_embeds_2 shape:',aux_embeds_2.shape)
              #print('cons det shape:',track_queries['cons_det'].shape)
              #l_dict = {'distance_track':cons_loss(aux_embeds_2.repeat_interleave(track_queries['cons_det'].shape[0],dim=0),track_queries['cons_det'].repeat_interleave(aux_embeds_2.shape[0],dim=0),aux_mask.flatten())}
              v_idxs, idxs = torch.max(ious,dim=0)
              idxs[v_idxs < 0.5] = -1
              cont = track_queries['cons_track'].shape[0]
              for i in range(len(idxs)):
                if idxs[i]==-1:
                  idxs[i]=cont
                  cont+=1
              #print('cons_track shape:',track_queries['cons_track'].shape)
              #print('cons_det shape:',track_queries['cons_det'].shape)
              aux_constrack = torch.cat((track_queries['cons_track'].clone(),track_queries['cons_det']),0)
              aux_idxs = torch.cat((torch.arange(0,track_queries['cons_track'].shape[0]).to('cuda'),idxs),0)
              #print('aux_idxs:',aux_idxs)
              l_dict = {'distance_track':self.cons_loss(aux_constrack,aux_idxs)}
              losses_dict.update(l_dict)
              
              res = F.normalize(track_queries['cons_track'] @ F.normalize(track_queries['cons_det']).t())
              #l_dict = {'res_track':F.mse_loss(res,ious)}
              binary_ious = ious > 0.5
              binary_ious = binary_ious.float()
              l_dict = {'res_track':F.binary_cross_entropy_with_logits(res,ious)}
              losses_dict.update(l_dict)
              
              """
              targets[0]['labels'] = targets[0]['labels'][indices_det]
              targets[0]['boxes'] = targets[0]['boxes'][indices_det]
              aux_track_queries = {'pred_logits':track_queries['pred_logits_2_b'],'pred_boxes':track_queries['pred_boxes_2_b']}
              indices = self.matcher(aux_track_queries,targets)
              for loss in self.losses:
                  l_dict = self.get_loss(loss, aux_track_queries, targets, indices, num_boxes)
                  l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                  l_dict = {k + f'_track': v for k, v in l_dict.items()}
                  losses_dict.update(l_dict)
              """
              
          """"
          #if 'pred_logits_2_b' in track_queries and 'det_idxs' in track_queries:
          if 'pred_logits_2_b' in track_queries:
              #print('len indices_det:',indices_det.shape)
              th = 0.5
              with torch.no_grad():
                scores = F.sigmoid(outputs_without_aux['pred_logits'])
                mask = torch.max(scores,2)[0] > th
              if mask.sum() > 0:
                ious = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['pred_boxes_2_b'].squeeze(0)),box_cxcywh_to_xyxy(outputs_without_aux['pred_boxes'][mask]))
                #ious_scores, ious_indexs = torch.topk(ious,10)
                ious_scores, ious_indexs = torch.max(ious,dim=-1)
                #mask_iou = ious_scores > 0.5
                mask_iou = ious_scores > th
                aux_mask = ious > th
                aux_mask = aux_mask.float()
                if mask_iou.sum() > 0 and aux_mask.sum() > 0:
                   #Probando la multi-asignacion
                   #ious = torch.topk(ious,10)[1]#esta parte se trae los mejores 10
                   #ious_mask = torch.zeros(track_queries['attn_weights'].shape).to(track_queries['attn_weights'].device)
                   #ious_mask[torch.nonzero(aux_mask)]=1
                   #ious_mask[:,ious]=1
                   #ious_mask[:,torch.nonzero(aux_mask)]=1
                   #l_dict = {"idxs_track":criterion(track_queries['attn_weights'][mask_iou],ious_mask[mask_iou])}
                   #pos_weight = torch.ones([ious_mask[mask_iou].shape[1]]).to(track_queries['attn_weights'].device)
                   #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                   #print('logits attn weights:',track_queries['attn_weights'][mask_iou])
                   
                   print('ious:',ious)
                   print('target:',aux_mask)
                   print('target:',aux_mask[mask_iou])
                   print('attn_weights shape:',track_queries['attn_weights'])
                   print('det_idxs shape:',track_queries['det_idxs'])
                   det_shape = outputs_without_aux['pred_boxes'][mask].shape[0]
                   #l_dict = {'idxs_track':F.binary_cross_entropy_with_logits(track_queries['attn_weights'][mask_iou],aux_mask[mask_iou])}
                   l_dict = {'idxs_track':F.binary_cross_entropy_with_logits(track_queries['det_idxs'][mask_iou][:,:det_shape],aux_mask[mask_iou])}
                   #l_dict = {"idxs_track":criterion(track_queries['attn_weights'][mask_iou],ious_mask[mask_iou])}
                   losses_dict.update(l_dict)
                   
                   print('revisando la atencion')
                   print('attn:',track_queries['attn_weights'])
                   print('ious:',ious)
                   l_dict = {'attn_loss': F.mse_loss(track_queries['attn_weights'],ious)}
                   losses_dict.update(l_dict)
                   #Probando minimizar la distancia de las queries
                   #print('embeds 2 shape:',track_queries['embeds_2'].squeeze(0)[mask_iou].shape)
                   #print('shape hs:',track_queries['hs'].shape)
                   #print('mask shape:',mask.shape)
                   #l_dict = {'distance_track':F.cosine_embedding_loss(track_queries['embeds_2_b'].squeeze(0)[mask_iou],track_queries['hs'][mask][ious_indexs][mask_iou],d_t)}
                   
                   aux_mask = torch.where(aux_mask > 0, 1.0, -1.0)
                   aux_embeds_2 = track_queries['embeds_2_b'].squeeze(0).clone()
                   #print('input 1 shape:',aux_embeds_2.repeat_interleave(track_queries['hs'][mask].shape[0],dim=0).shape)
                   #print('input 2 shape:',track_queries['hs'][mask].repeat_interleave(aux_embeds_2.shape[0],dim=0).shape)
                   #print('targets:',aux_mask.flatten().shape) volver a colocar
                   l_dict = {'distance_track':F.cosine_embedding_loss(aux_embeds_2.repeat_interleave(track_queries['hs'][mask].shape[0],dim=0),track_queries['hs'][mask].repeat_interleave(aux_embeds_2.shape[0],dim=0),aux_mask.flatten())}
                   losses_dict.update(l_dict)
                   #Match trackqueries with dets
                   mask_det = torch.zeros(outputs_without_aux['pred_logits'].shape[1]).to(outputs_without_aux['pred_logits'].device)
                   mask_det[torch.unique(ious_indexs)] = 1
                   mask_det = mask_det > 0
                   aux_track_queries = {'pred_logits':track_queries['pred_logits_2_b'],'pred_boxes':track_queries['pred_boxes_2_b']}

                   with torch.no_grad():
                        det_target_classes = torch.argmax(torch.sigmoid(outputs_without_aux['pred_logits'].squeeze(0)[mask_det]),dim=-1)

                   aux_det_targets = [{'labels':det_target_classes,'boxes':outputs_without_aux['pred_boxes'].squeeze(0)[mask_det]}]
                   indices = self.matcher(aux_track_queries,aux_det_targets)
                   for loss in self.losses:
                      l_dict = self.get_loss(loss, aux_track_queries, aux_det_targets, indices, num_boxes)
                      l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                      l_dict = {k + f'_inter': v for k, v in l_dict.items()}
                      losses_dict.update(l_dict)

                #Calcular el error de los tracks contra los gt que han coincidido con las dets
                targets[0]['labels'] = targets[0]['labels'][indices_det]
                targets[0]['boxes'] = targets[0]['boxes'][indices_det]
                aux_track_queries = {'pred_logits':track_queries['pred_logits_2_b'],'pred_boxes':track_queries['pred_boxes_2_b']}
                indices = self.matcher(aux_track_queries,targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_track_queries, targets, indices, num_boxes)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_track': v for k, v in l_dict.items()}
                    losses_dict.update(l_dict)
          """    
          return losses_dict, track_queries
