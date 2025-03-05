from .rtdetr_criterion import SetCriterion
from src.core import register
import torch
from src.misc.dist import get_world_size, is_dist_available_and_initialized
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
#import torchvision

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        #print("euclidean distance:",euclidean_distance)
        pos = (1-label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive
"""
def match_by_iou(track_boxes,det_boxes):
    print('track_boxes shape:',track_boxes.shape)
    print('det_boxes shape:',det_boxes.shape)
    return torchvision.ops.generalized_box_iou(track_boxes,det_boxes)
"""

@register
class CriterionMOT_v2(SetCriterion):
      def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e4, num_classes=80):
          super().__init__(matcher, weight_dict, losses,alpha, gamma, eos_coef, num_classes)
          self.losses = losses
          self.cons_loss = ContrastiveLoss()
          self.cons_id_loss = nn.CrossEntropyLoss()

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
          #print('pred logits det:',outputs_without_aux['pred_logits'])
          indices = self.matcher(outputs_without_aux, targets)
          for loss in self.losses:
              l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
              l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
              losses_dict.update(l_dict)

          if 'aux_outputs' in outputs:
             for i, aux_outputs in enumerate(outputs['aux_outputs']):
               indices = self.matcher(aux_outputs, targets)
               for loss in self.losses:
                    #print("loss a calcular:",loss)
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses_dict.update(l_dict)

          #Track loss
          if mode!='cons' and 'pred_logits' in track_queries and track_queries['pred_logits'] is not None and track_queries['pred_logits'].shape[1]>0:
             #aux_mask = torch.max(outputs_without_aux['pred_logits'],2)[0] > 0.2
             #print('outputs det:',outputs_without_aux['pred_boxes'][aux_mask])
             #print('targets:',targets)
             #print('track_queries shape:',track_queries['pred_boxes'].shape)
             #print('track_queries:',track_queries['pred_boxes'])
             #cross_entropy = nn.CrossEntropyLoss()
             indices = self.matcher(track_queries,targets)
             for loss in self.losses:
                 l_dict = self.get_loss(loss, track_queries, targets, indices, num_boxes)
                 l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                 l_dict = {k + f'_track': v for k, v in l_dict.items()}
                 #print('track loss:',l_dict)
                 losses_dict.update(l_dict)
                 #l_dict = {'constant_track': 0} #Inicialmente estaba en 100
                 #losses_dict.update(l_dict)
                 #l_dict = {'add_keep_loss_track':cross_entropy(len(indices),len(targets))}
                 #losses_dict.update(l_dict)
             
             #loss new decoder
             if 'attn_weights' in track_queries:
                #print("embeds_b shape:",track_queries["embeds_b"].shape)
                #print('track queries boxes shape:',track_queries['pred_boxes'].shape)
                aux_track_queries = [{'labels':torch.argmax(outputs_without_aux['pred_logits'],dim=-1).squeeze(0),'boxes':outputs_without_aux['pred_boxes'].squeeze(0)}]
                #print("pred_logits_b shape:",track_queries['track_dec_out_logits_b'].shape)
                #print("pred_boxes_b shape:",track_queries['track_dec_out_bboxes_b'].shape)
                #print('aux labels shape:',aux_track_queries[0]['labels'].shape)
                aux_track_queries_pred = {"pred_logits":track_queries['track_dec_out_logits_b'],"pred_boxes":track_queries['track_dec_out_bboxes_b']}
                #print('logits shape:',aux_track_queries_pred['pred_logits'].shape)
                indices = self.matcher(aux_track_queries_pred, aux_track_queries)
                #print('indices new decoder:',indices)
                #print('attn_weigths:',track_queries['attn_weights'])
                indexs_attn = torch.argmax(track_queries['attn_weights'],dim=-1).squeeze(0)
                #max_attn_top, max_idxs_det_top = torch.topk(track_queries['attn_weights'], 4)
                #indexs_attn = indexs_attn[indices[0][0]].float()
                #indexs_attn.requires_grad=True
                #indexs_attn = indexs_attn.type(torch.int64)
                #print('argmax attn:',torch.argmax(track_queries['attn_weights'],dim=-1).shape)
                #print('argmax attn:',torch.argmax(track_queries['attn_weights'],dim=-1))
                #print('attn_weights shape:',track_queries['attn_weights'].shape)
                #print('entrada 1:',track_queries["attn_weights"].shape)
                #print("entrada 2:",indices[0][1].shape)
                #print("pred_boxes shape:",outputs_without_aux['pred_boxes'].shape)
                #print("cajas pred select by attn:",outputs_without_aux['pred_boxes'][:,indexs_attn])
                #print("cajas pred select by hungarian:",outputs_without_aux['pred_boxes'][:,indices[0][1]])
                #l_dict = {'index_loss':F.l1_loss(indexs_attn, indices[0][1].float().to(track_queries['attn_weights'].device), reduction='none').sum()}
                #match_iou = match_by_iou(track_queries['track_dec_out_bboxes_b'].squeeze(0),aux_track_queries[0]['boxes'])
                #print('match_iou:',match_iou)
                ious = generalized_box_iou(box_cxcywh_to_xyxy(track_queries['track_dec_out_bboxes_b'].squeeze(0)),box_cxcywh_to_xyxy(aux_track_queries[0]['boxes']))
                #print("ious porcentajes:",torch.topk(ious,10)[0])
                ious = torch.topk(ious,10)[1]
                #print("cajas track:",box_cxcywh_to_xyxy(track_queries['track_dec_out_bboxes_b'].squeeze(0)))
                #print("cajas deteccion:",box_cxcywh_to_xyxy(aux_track_queries[0]['boxes']))
                #print('ious:',ious)
                #print('indices:',indices)
                #print('indciers 1:',indices[0][1])
                #print('cajas de track:',box_cxcywh_to_xyxy(track_queries['track_dec_out_bboxes_b'].squeeze(0)))
                #print('revisar:',box_cxcywh_to_xyxy(aux_track_queries[0]['boxes']).shape)
                #print('cajas de deteccion que seleccionan los indices:',box_cxcywh_to_xyxy(aux_track_queries[0]['boxes'])[indices[0][1]])
                ious_mask = torch.zeros(track_queries['det_idxs'].shape).to(track_queries['det_idxs'].device)
                ious_mask[:,ious]=1
                #print('ious one_hot:',ious_mask.shape)
                #print('det_idxs shape:',track_queries['det_idxs'].shape)
                #with torch.no_grad():
                #     max_attn_top, max_idxs_det_top = torch.topk(torch.sigmoid(track_queries['det_idxs']), 10)
                     #print("max_idxs_det_top:",max_idxs_det_top)
                #print('ious shape:',ious.shape)
                #print('attn shape:',track_queries["attn_weights"].shape)
                #l_dict = {"index_loss":F.nll_loss(torch.log(track_queries["attn_weights"].squeeze(0) + 1e-20), indices[0][1].to(track_queries['attn_weights'].device))}
                #l_dict = {"index_loss":F.nll_loss(torch.log(track_queries["attn_weights"].squeeze(0) + 1e-20), ious.to(track_queries['attn_weights'].device))}
                if 'ids_det' in track_queries:
                   print('track boxes:',box_cxcywh_to_xyxy(track_queries['track_dec_out_bboxes_b'].squeeze(0)))
                   print('det boxes:',box_cxcywh_to_xyxy(aux_track_queries[0]['boxes'])[track_queries['ids_det']])
                   print('iou prueba:',generalized_box_iou(box_cxcywh_to_xyxy(track_queries['track_dec_out_bboxes_b'].squeeze(0)),box_cxcywh_to_xyxy(aux_track_queries[0]['boxes'][track_queries['ids_det']].squeeze(0))))
                   print('iou prueba invertido:',generalized_box_iou(box_cxcywh_to_xyxy(aux_track_queries[0]['boxes'][track_queries['ids_det']].squeeze(0)),box_cxcywh_to_xyxy(track_queries['track_dec_out_bboxes_b'].squeeze(0))))
                pos_weight = torch.ones([300]).to(track_queries['attn_weights'].device)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                #print('pasa criterion')
                #l_dict = {"index_loss":F.nll_loss(max_idxs_det_top, ious.to(track_queries['attn_weights'].device))}
                #print("det_idxs:",track_queries['det_idxs'])
                #print("ious_mask:",torch.nonzero(ious_mask))
                #l_dict = {"index_loss":criterion(track_queries['det_idxs'],ious_mask)*3}
                #print("attn_weights shape:",track_queries['attn_weights'].shape)
                #print("ious_mask shape:",ious_mask.shape)
                l_dict = {"index_loss":criterion(track_queries['attn_weights'],ious_mask)*3}
                #print("index_loss:",l_dict["index_loss"])
                #l_dict['index_loss'] /= indexs_attn.shape[0]
                #print("index_attn shape:",indexs_attn.shape)
                #print('loss_index:',l_dict['index_loss'])
                losses_dict.update(l_dict)
                """
                attn_idxs = torch.tensor(track_queries["det_idxs"].shape[0]]).float().to(track_queries['attn_weights'].device)
                attn_idxs.requires_grad=True
                l_dict = {'gt_loss':F.l1_loss(attn_idxs,torch.tensor([num_boxes]).float().to(track_queries['attn_weights'].device))}
                losses_dict.update(l_dict)
                """
          """
          if mode!='cons' and 'pred_logits' in track_queries and track_queries['pred_logits'] is None:
             l_dict = {'constant_track': 10}
             losses_dict.update(l_dict)
          """
          #cons
          if "embeds_cons" in track_queries and track_queries['embeds_cons'] is not None and mode=='cons':
             #cosine_distance = nn.PairwiseDistance(p=2)
             #print('track_queries:',track_queries)
             cross_entropy = nn.CrossEntropyLoss()
             #print('embeds_cons shape:',track_queries['embeds_cons'].shape)
             #print('det_queries shape:',track_queries['det_queries'].shape)
             output_d = pairwise_cosine_similarity(track_queries['embeds_cons'],track_queries['det_queries']) # get indexs
             #print('output_d:',output_d)
             #print('output_d shape:',output_d.shape)
             print('det_boxes:',track_queries['det_boxes'])
             print('track_boxes:',track_queries['track_boxes'])
             positives = []
             negatives = []
             for i, d in enumerate(output_d):
                 positives.append([i,torch.argmax(d)])
                 if torch.argmin(d) != torch.argmax(d):
                    negatives.append([i,torch.argmin(d)])
                 else:
                    negatives.append([])
             #print('positives:',positives)
             cons_loss = []
             for v0,v1 in zip(positives,negatives):
                 cons_pos = self.cons_loss(track_queries["track_proj"][v0[0]],track_queries["det_proj"][v0[1].item()],0) #calculate contrastive loss for positives
                 #print("cons_pos:",cons_pos)
                 if len(v1)>0:
                    cons_neg = self.cons_loss(track_queries["track_proj"][v1[0]],track_queries["det_proj"][v1[1].item()],1) #calculate contrastive loss for negatives
                 else:
                    cons_neg = 0
                 #print("cons_neg:",cons_neg)
                 cons_loss.append(cons_pos+cons_neg)
             cons_loss = torch.tensor(cons_loss).sum()/len(cons_loss)
             #sum losses
             #id_loss = self.cons_id(track_queries["proj_cons"],ids_det)
             l_dict = {}
             l_dict['loss_ids_cons'] = cross_entropy(track_queries['det_ids'],torch.tensor(positives)[:,1].to(track_queries['det_ids'].device))
             #print('loss_ids_cons:',l_dict['loss_ids_cons'])
             losses_dict.update(l_dict)
             l_dict = {}
             l_dict['loss_cons_proj'] = cons_loss
             losses_dict.update(l_dict)
          """
          #track process
          th = 0.5
          mask = torch.max(outputs_without_aux['pred_logits'],2)[0] > th
          det_embeds = track_queries['hs'][-1][mask]
          ref_points = track_queries['ref_points'][-1][mask]
          pos_encoding = track_queries['pos_encoding'][-1][mask]
          if track_queries['embeds'] is None:
             if mask.sum()>0:
                track_queries['embeds'] = det_embeds.unsqueeze(0)
                track_queries['track_ref_points'] = ref_points.unsqueeze(0)
                track_queries['track_pos_encoding'] = pos_encoding.unsqueeze(0)
                #print('agrega primeros:',track_queries['embeds'].shape)
          else:
             idxs_mask = torch.argmax(track_queries['keep'],dim=2)
             keep_mask = idxs_mask == 1
             idxs_mask_dets = torch.argmax(track_queries['add'],dim=2)
             add_mask = idxs_mask_dets == 1
             add_mask = add_mask.squeeze(0)
             aux_pred_boxes = torch.cat((track_queries['track_dec_out_bboxes'][keep_mask],outputs_without_aux["pred_boxes"][track_queries['add_mask']][add_mask]))
             aux_pred_boxes = aux_pred_boxes.unsqueeze(0)
             aux_pred_logits = torch.cat((track_queries['track_dec_out_logits'][keep_mask],outputs_without_aux["pred_logits"][track_queries['add_mask']][add_mask]))
             aux_pred_logits = aux_pred_logits.unsqueeze(0)

             aux_track_queries = {'pred_boxes':aux_pred_boxes,'pred_logits':aux_pred_logits}

             #delete queries:
             track_queries['embeds'] = track_queries['embeds'][keep_mask].unsqueeze(0) #probar
             track_queries['track_ref_points'] = track_queries['track_ref_points'][keep_mask].unsqueeze(0)
             track_queries['track_pos_encoding'] = track_queries['track_pos_encoding'][keep_mask].unsqueeze(0)

             #loss for track queries:
             if aux_pred_boxes.shape[1] > 0:
                indices = self.matcher(aux_track_queries,targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_track_queries, targets, indices, num_boxes)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_track': v for k, v in l_dict.items()}
                    #print('track loss:',l_dict)
                    losses_dict.update(l_dict)
                    l_dict = {'constant_track': 0} #Inicialmente estaba en 100
                    losses_dict.update(l_dict)
             else:
               l_dict = {'constant_track': len(targets)} #Inicialmente estaba en 100
               losses_dict.update(l_dict)
             #print('loss when use track:',losses_dict['loss_bbox_track'])
          """
          return losses_dict, track_queries
