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

          if track_queries is not None:
             if track_queries['embeds'] is None:
                mask = output_without_aux['pred_logits']>0.5
                track_queries['embeds'] = track_queries['hs'][mask].clone()
             else:
                cosine_loss = nn.CosineEmbeddingLoss()
                idxs_max = torch.argmax(track_queries['ids'],dim=2)
                print('idxs_max':idxs_max)
                target = torch.ones(len(idxs_max))
                sim_loss = loss(track_queries['embeds'], , target)
                losses_dict.update({'sim':sim_loss})
                mask = output_without_aux['pred_logits']>0.5
                track_queries['embeds'] = torch.cat((track_queries['embeds'],track_queries['hs'][mask]))

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
