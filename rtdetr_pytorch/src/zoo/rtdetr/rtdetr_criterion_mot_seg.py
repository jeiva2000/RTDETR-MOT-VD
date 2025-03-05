from .rtdetr_criterion import SetCriterion
from src.core import register
import torch
from src.misc.dist import get_world_size, is_dist_available_and_initialized
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from pytorch_metric_learning import losses as l_c
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
import cv2

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    prob = prob.flatten()
    inputs = inputs.flatten()
    targets = targets.flatten()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    #return loss.mean(1).sum()# / num_boxes
    #return loss.mean(1).sum() / num_boxes
    return loss.mean()

def dice_loss(inputs, targets, smooth=1e-6, gamma=2):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten()
    targets = targets.flatten()
    numerator = 2 * (inputs * targets).sum() + smooth
    denominator = (inputs**gamma).sum() + (targets**gamma).sum() + smooth
    loss = 1 - (numerator) / (denominator)
    return loss
    #return loss.sum() / num_boxes

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

      def match(self, outputs, track_queries, targets, image, th_filter=0.5, mode=''):
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
          
          #clustering
          pred_boxes = box_cxcywh_to_xyxy(outputs_without_aux['pred_boxes'].squeeze(0)[indices_pred])
          seg_boxes = outputs_without_aux['seg_boxes']
          image = image.squeeze(0)#.permute(1,2,0)
          image_zeros = torch.zeros(image.shape[1:3],requires_grad=True).to('cuda')
          image_labels = torch.zeros(image.shape[1:3]).to('cuda')
          segs_c = [] 
          labels_c = []
          for pred, seg in zip(pred_boxes,seg_boxes):
            pred = pred * 640
            xmin, ymin, xmax, ymax = int(pred[0].item()), int(pred[1].item()), int(pred[2].item()), int(pred[3].item())
            #postprocess
            crop = image[:,ymin:ymax,xmin:xmax]
            crop_aux = crop.clone()
            crop = crop.reshape(crop.shape[0],crop.shape[1]*crop.shape[2]).permute(1,0)
            crop = crop.cpu().numpy()
            if crop.shape[0] == 0:
              continue
            kmeans = KMeans(n_clusters=2,n_init=10).fit(crop)
            labels = torch.tensor(kmeans.labels_).unsqueeze(0)
            labels = labels.reshape(crop_aux.shape[1],crop_aux.shape[2])
            #labels = labels > 0     
            #labels = ~labels
            #seg = torch.nn.functional.pad(seg,(0,1,0,1),'constant',0)
            #seg = seg.reshape(image_zeros.shape)
            #print('crop shape:',crop_aux.shape)
            seg = cv2.resize(seg.permute(1,2,0).cpu().detach().numpy(),(crop_aux.shape[2],crop_aux.shape[1]))
            #print('seg shape:',seg.shape)
            seg = torch.tensor(seg).to('cuda')
            image_zeros[ymin:ymax,xmin:xmax] += seg
            image_labels[ymin:ymax,xmin:xmax] = labels

          #l_dict = {'seg_loss':torch.nn.functional.binary_cross_entropy_with_logits(image_zeros,image_labels)}
          #losses_dict.update(l_dict)

          #l_dict = {'seg_loss_focal':sigmoid_focal_loss(image_zeros,image_labels,num_boxes)}
          #losses_dict.update(l_dict)

          #cv2.imwrite('prueba_seg/prueba_kmeans.png',image_labels.cpu().detach().numpy()*255)
          #cv2.imwrite('prueba_seg/prueba_seg.png',image_zeros.cpu().detach().numpy()*255)

          #l_dict = {'dice_loss':dice_loss(image_zeros,image_labels)}
          #losses_dict.update(l_dict)

          if 'k_mask' not in track_queries:
            track_queries['k_mask'] = image_labels.unsqueeze(2)
          else:
            track_queries['k_mask'] = torch.cat((track_queries['k_mask'],image_labels.unsqueeze(2)),dim=-1)
          if 's_mask' not in track_queries:
            track_queries['s_mask'] = image_zeros.unsqueeze(2)
          else:
            track_queries['s_mask'] = torch.cat((track_queries['s_mask'],image_zeros.unsqueeze(2)),dim=-1)
            
          return losses_dict, track_queries

      def calc_seg_loss(self, image, losses_dict, track_queries):
        #print('k_mask shape concat:',track_queries['k_mask'].shape)
        k_mask = track_queries['k_mask'].mean(2)
        s_mask = track_queries['s_mask'].mean(2)
        #print('k_mask shape mean:',k_mask.shape)
        #print('s_mask shape mean:',s_mask.shape)
        cv2.imwrite('prueba_seg/prueba_kmeans.png',k_mask.cpu().detach().numpy()*255)
        aux_s_mask = torch.sigmoid(s_mask)
        aux_s_mask = (aux_s_mask > 0.5).float()
        cv2.imwrite('prueba_seg/prueba_seg.png',aux_s_mask.cpu().detach().numpy()*255)
        l_dict = {'dice_loss':dice_loss(s_mask,k_mask)}
        losses_dict.update(l_dict)
        #print('image shape:',image.shape)
        cv2.imwrite('prueba_seg/orig_image.png',image.squeeze(0).permute(1,2,0).cpu().detach().numpy()*255)
        return losses_dict


