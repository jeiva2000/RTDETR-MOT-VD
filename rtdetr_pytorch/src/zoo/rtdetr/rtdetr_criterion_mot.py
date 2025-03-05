from .rtdetr_criterion import SetCriterion
from src.core import register
import torch
from src.misc.dist import get_world_size, is_dist_available_and_initialized

@register
class CriterionMOT(SetCriterion):
      def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e4, num_classes=80):
          super().__init__(matcher, weight_dict, losses,alpha, gamma, eos_coef, num_classes)
          self.losses = losses
 
      def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

      def match(self, outputs, track_queries, targets, th_filter=0.5):
          losses_dict = {}

          num_boxes = sum(len(t["labels"]) for t in targets)
          num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
          if is_dist_available_and_initialized():
             torch.distributed.all_reduce(num_boxes)
          num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

          outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
          #print("outputs_with_aux:",outputs_without_aux['pred_logits'].shape)
          indices = self.matcher(outputs_without_aux, targets)
          for loss in self.losses:
              l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
              l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
              losses_dict.update(l_dict)
          #print("pasa first losses")
          if 'aux_outputs' in outputs:
             for i, aux_outputs in enumerate(outputs['aux_outputs']):
               #print("aux_outputs:",aux_outputs)
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
                    #print("calcula loss:",loss)
          """
          #Get ids and new objects:
          th_score = 0.5
          pred_ids = outputs_without_aux['pred_ids']
          #print('pred_ids shape:',pred_ids.shape)
          values, indices = torch.max(pred_ids,dim=2)
          #print('indices:',indices)
          #print('nuevas consultas:',indices==new_idx)
          new_mask = indices==1
          #print('sum new_mask 1',new_mask.sum())
          if new_mask.sum() > 0:
             new_queries = track_queries['hs'][-1][new_mask].unsqueeze(0)
             if track_queries['track_queries'] is not None:
                track_queries['track_queries'] = torch.concat((track_queries['track_queries'],new_queries),1)
             else:
                track_queries['track_queries'] = new_queries
             print('agregaa')
             #print('track_query shape:',track_queries['track_queries'].shape)

          #loss for track queries:
          aux_track_queries = {'pred_boxes':track_queries['pred_boxes'],'pred_logits':track_queries['pred_logits']}
          indices = self.matcher(outputs_without_aux,targets)
          for loss in self.losses:
              l_dict = self.get_loss(loss, aux_track_queries, targets, indices, num_boxes)
              l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
              l_dict = {k + f'_track_{i}': v for k, v in l_dict.items()}
              #print('track loss:',l_dict)
              losses_dict.update(l_dict)
          """ #volver a colocar codigo de arriba
          """
          #match with track queries
          aux_track_queries = {'boxes':track_queries['pred_boxes'],'labels':track_queries['pred_logits']}
          dect_idxs = torch.arange(len(output_without_aux))
          indices = self.matcher(outputs_without_aux,aux_track_queries)
          print('indices:',indices)
          unmatched_dect_idxs = torch.any(dect_idxs != torch.Tensor(indices[0]), axis=1).nonzero().flatten()
          mask = torch.max(outputs_without_aux['pred_logits'][unmatched_dect_idxs],dim=2)[0]>th_filter
          track_queries['track_queries'] = torch.stack(track_queries['track_queries'],track_queries['hs'][-1][mask])
          """
          #indices[1]
          #filter outputs:
          """
          th_score = 0.5
          pred_logits = outputs['pred_logits']
          pred_boxes = outputs['pred_boxes']
          mask = torch.max(pred_logits,dim=2)[0]>th_score
          if mask.sum()>0:
             #print('track_queries shape:',out_track['track_queries'].shape)
             out_track['track_queries'] = out_track['hs'][-1][mask]
             h_points = out_track['h_points']
             print('h_points:',h_points)
             print("det outputs shape:",out_track["hs"][-1][mask].shape)
          """
          """
          if outputs_2 is not None:
             outputs_without_aux = {k: v for k, v in outputs_2.items() if 'aux' not in k}
             #filter predictions
             th_score = 0.5
             pred_logits = outputs['pred_logits']
             pred_boxes = outputs['pred_boxes']
             #print(torch.max(pred_logits,dim=1)[0])
             #print('pred_logits before:',pred_logits.shape)
             mask = torch.max(pred_logits,dim=2)[0]>th_score
             #print('mask:',mask.shape)
             pred_logits = pred_logits[mask]
             #print('pred_logits after:',pred_logits.shape)
             pred_boxes = pred_boxes[mask]
             det_queries = {'labels':pred_logits, 'boxes':pred_boxes}
             #print('pasa det_queries')
             if det_queries['labels'].shape[0]>0:
                #print('pasa algoo')
                #print('det_queries:',det_queries)
                det_queries['labels'] = torch.argmax(det_queries['labels'],dim=1)
                #print('det_queries:',det_queries) 
                first_indices = self.matcher(outputs_without_aux,[det_queries])
                #print('first_indices:',first_indices[0][1])
                unmatch_indices = [idx for idx in range(len(pred_logits)) if idx not in first_indices[0][1]]
                #print('unmatch_indices:',unmatch_indices)
             indices = self.matcher(outputs_without_aux, targets)
             for loss in self.losses:
                 l_dict = self.get_loss(loss, outputs_without_aux, targets, indices, num_boxes, **kwargs)
                 l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                 l_dict = {k + f'_last_{i}': v for k, v in l_dict.items()}
                 losses_dict.update(l_dict)
          """
          return losses_dict, track_queries
      """
      def forward(self, outputs, input_data: dict):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses
      """
