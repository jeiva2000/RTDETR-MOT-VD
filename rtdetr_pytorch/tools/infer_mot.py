import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image, ImageDraw
import sys
sys.path.append("..")
from src.core import YAMLConfig
import argparse
from pathlib import Path
import time
import os
import shutil
import copy 
from pickle import dump
import math
import cv2
from sklearn.cluster import KMeans
import numpy as np
import copy
from torchvision.ops.boxes import box_area


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


class ImageReader:
    def __init__(self, resize=(224,224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
             transforms.Resize((resize, resize)) if isinstance(resize, int) else transforms.Resize(
                 (resize[0], resize[1])),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ])
        self.resize = resize
        self.pil_img = None   #保存最近一次读取的图片的pil对象

    def __call__(self, image_path, *args, **kwargs):
        """
        读取图片
        """
        self.pil_img = Image.open(image_path).convert('RGB')
        old_size = self.pil_img.size
        #self.pil_img = self.pil_img.resize((self.resize[0], self.resize[1]))
        return old_size, self.transform(self.pil_img).unsqueeze(0)

class Model(nn.Module):
    def __init__(self, confg=None, ckpt="") -> None:
        super().__init__()
        self.cfg = YAMLConfig(confg, resume=ckpt)
        if ckpt:
            checkpoint = torch.load(ckpt, map_location='cpu') 
            if 'ema' in checkpoint:
                state = checkpoint['ema']['module']
            else:
                state = checkpoint['model']
        else:
            raise AttributeError('only support resume to load model.state_dict by now.')

        # NOTE load train mode state -> convert to deploy mode
        self.cfg.model.load_state_dict(state, strict=False)
        #self.cfg.model.load_state_dict(state)

        self.model = self.cfg.model.deploy()
        self.postprocessor = self.cfg.postprocessor.deploy()
        # print(self.postprocessor.deploy_mode)
        
    def forward(self, images, track_queries, orig_target_sizes):
        #print('len images:',len(images))
        #outputs_det, track_queries = self.model.forward_inference_2(images,track_queries,ind_track=False)
        #print('imagen a enviar a inference 2:',images)
        outputs_det, track_queries = self.model.forward_inference_2(images,track_queries,ind_track=True)
        det_embeds = outputs_det['embeds']
        if 'ids' in track_queries:
          ids = track_queries['ids']
        else:
          ids = []
        if "delete" in track_queries:
           delete = track_queries["delete"]
        else:
           delete = []
        self.postprocessor.num_top_queries = outputs_det['pred_boxes'].shape[1]
        if 'seg_boxes' in outputs_det:
          segs = outputs_det['seg_boxes']
        else:
          segs = None
        if 'pred_masks' in outputs_det:
          pred_masks = outputs_det['pred_masks']
          #print('pred masks shape:',pred_masks.shape)
          #print('orig target sizes:',orig_target_sizes.shape)
          pred_masks *=orig_target_sizes
        else:
          pred_masks = None
        outputs_det = self.postprocessor(outputs_det, orig_target_sizes)


        if 'pred_boxes_2' not in track_queries or track_queries['pred_boxes_2'] is None:
           outputs_track = None
           aux_outputs_track = None
        else:
           #print('track queries boxes:',track_queries['pred_boxes_2'])
           self.postprocessor.num_top_queries = track_queries['pred_boxes_2'].shape[1]
           track_queries['pred_logits'] = track_queries['pred_logits_2']
           track_queries['pred_boxes'] = track_queries['pred_boxes_2']
           print('orig target size:',orig_target_sizes)
           print('boxes track before process:',track_queries['pred_boxes'])
           outputs_track = self.postprocessor(track_queries, orig_target_sizes,ind_track=True)
           if 're_boxes' in track_queries and track_queries['re_boxes'] is not None and track_queries['re_boxes'].shape[1]>0:
            #print('re_logits post:',track_queries['re_logits'])
            #print('re_boxes post:',track_queries['re_boxes'])
            aux_track_queries = {'pred_logits':track_queries['re_logits'],'pred_boxes':track_queries['re_boxes']}
            self.postprocessor.num_top_queries = track_queries['re_boxes'].shape[1]
            aux_outputs_track = self.postprocessor(aux_track_queries, orig_target_sizes, ind_track=True)
            #print('aux_outputs_track:',aux_outputs_track)
           else:
            aux_outputs_track = None

        if 'boxes' in track_queries and track_queries['boxes'].shape[1] > 0:
           track_queries['pred_logits'] = track_queries['logits']
           track_queries['pred_boxes'] = track_queries['boxes']
           self.postprocessor.num_top_queries = track_queries['boxes'].shape[1]
           outputs_track = self.postprocessor(track_queries, orig_target_sizes,ind_track=True)

        return det_embeds, outputs_det, outputs_track, aux_outputs_track, ids, track_queries, delete, segs, pred_masks

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/serperzar/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r101vd_6x_mot_coco_v3.yml", help="配置文件路径")
    parser.add_argument("--ckpt", default="../rtdetr_101vd_6x_mot_track_v3_prueba_revision/best_checkpoint.pth", help="权重文件路径")
    parser.add_argument("--video_path", default="/data/backup/serperzar/mot/dataset_mot_damages_v5/test", help="待推理图片路径")
    parser.add_argument("--output_dir", default="/home/serperzar/RT-DETR/rtdetr_pytorch/videos_output", help="输出文件保存路径")
    parser.add_argument("--device", default="cuda")

    return parser


def main(args):
    torch.cuda.memory._record_memory_history(True)
    folders = os.listdir(args.video_path)
    device = torch.device(args.device)
    #reader = ImageReader(resize=640)
    reader = ImageReader(resize=(640,640))
    #model = Model(confg=args.config, ckpt=args.ckpt)
    #model.model.init_tracks()
    #model.to(device=device)
    thrh = 0.5 #0.50 #0.20 #0.50 #0.3 normalmente debe ir en 0.50
    thrh_track = 0.30 #se usa 0.3 o 0.5  #0.001 #0.50 #0.3
    if os.path.exists(args.output_dir):
       shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)
    max_time = 0
    pred_dicts = {}
    for folder in folders:
        #track_queries = {'embeds':None,'ids':None}
        track_queries = {}
        model = Model(confg=args.config, ckpt=args.ckpt)
        #model.model.init_tracks()
        model.to(device=device)
        model.eval()
        predictions = []
        #print("folder:",folder)
        os.mkdir(os.path.join(args.output_dir,folder))
        images = os.listdir(os.path.join(args.video_path,folder))
        images = [name for name in images if name.endswith(('PNG','jpg'))]
        images.sort()
        pred_dicts[folder]={}
        #print('images:',images)
        for j, img_name in enumerate(images):
            """
            if j%10 == 0:
               print("inicia secuencia de 10:")
               model.model.init_tracks()
            """
            """
            if track_queries["embeds"] is not None:
               print("size embeds:",track_queries["embeds"].shape)
               print("size hs:",track_queries["hs"].shape)
               if "embeds_b" in track_queries:
                  print("size embeds_b:",track_queries["embeds_b"].shape)
               print("keys:",track_queries.keys())
            """
            old_size, img = reader(os.path.join(args.video_path,folder,img_name))
            #print('img orig size:',old_size)
            img = img.to(device)
            #print('img send shape:',img.shape)
            size = torch.tensor([[old_size[0], old_size[1]]]).to(device)
            #start = time.time()
            print('image_name:',img_name)
            #print('j:',j)
            det_embeds, output_det, output_track, aux_outputs_track, ids, track_queries, delete, segs, pred_masks = model(img, track_queries, size)
            print('output_det shape:',output_det[1].shape)
            pred_dicts[folder][img_name] = {'embeds':det_embeds.detach().cpu().numpy(),'ious':generalized_box_iou(output_det[1].squeeze(0),output_det[1].squeeze(0)).detach().cpu().numpy(),
            'boxes':output_det[1].detach().cpu().numpy(),'scores':output_det[2].detach().cpu().numpy()}
            print('output_track:',output_track)
            #print('output_track:',output_track)
            #print('segs:',segs)
            #print('output_det:',output_det)
            """
            print('boxes shape:',track_queries['pred_boxes'].shape)
            print('logits shape:',track_queries['pred_logits'].shape)
            mask = track_queries['pred_logits'] > thrh
            print('mask sum:',mask.sum())
            """
            """
            track_queries['pred_boxes'] = track_queries['pred_boxes'][mask]
            track_queries['pred_logits'] = track_queries['pred_logits'][mask]
            track_queries['hs'] = track_queries['hs'][mask]
            track_queries['embeds'] = track_queries['embeds'][mask]
            track_queries['track_ref_points'] = track_queries['track_ref_points'][mask]
            """
            #print(torch.cuda.memory_summary())
            #if j%10:
                #torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
                #snapshot = torch.cuda.memory._snapshot()
                #dump(snapshot, open('snapshot_{}.pickle'.format(str(j)), 'wb'))
                #Stop recording memory snapshot history.
                #torch.cuda.memory._record_memory_history(enabled=None)
            
            #ids = ids.cpu().detach().numpy()
            #print("ids:",ids)
            #aquii
            
            im = reader.pil_img
            im = im.resize((old_size[0], old_size[1]))
            im_aux = np.array(im)
            #print('im shape:',im_aux.shape)
            draw = ImageDraw.Draw(im)
            #print('track boxes:',boxes)
            #print('scores:',scores)
            labels, boxes, scores = output_det
            count_boxes = 0
            img_seg = torch.zeros((im_aux.shape[0],im_aux.shape[1],1))
            #img_seg = torch.zeros((im_aux.shape[1],im_aux.shape[0],1))
            img_seg_labels = torch.zeros((im_aux.shape[0],im_aux.shape[1])).to(boxes.device)
            for i, box in enumerate(boxes[0]):
                box = box.cpu().detach().numpy()
                scr = scores[0][i].cpu().detach().numpy()
                #print('scr:',scr)
                if scr >= thrh:
                   print("score det:",scr)
                   print("box:",box)
                   xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                   draw.rectangle(list(box), outline='blue',)
                   #draw.text((box[0], box[1]-30), text=str(scr), fill='blue', )
                   count_boxes+=1

                   if segs is not None:
                     crop = im_aux[ymin:ymax,xmin:xmax]
                     crop_aux = copy.deepcopy(crop)
                     #print('crop shape:',crop.shape)
                     if crop.shape[0] == 0 or crop.shape[1] == 0:
                        continue
                     crop = crop.reshape(crop.shape[0]*crop.shape[1],crop.shape[2])
                     kmeans = KMeans(n_clusters=2,n_init=10).fit(crop)
                     labels = torch.tensor(kmeans.labels_).unsqueeze(0)
                     labels = labels > 0
                     labels = ~labels
                     labels = labels.reshape(crop_aux.shape[0],crop_aux.shape[1])
                     #print('shape labels como sale:',labels.shape)
                     #print("img_seg:",img_seg.shape)
                     aux_seg = segs[i].clone()
                     aux_seg = torch.sigmoid(aux_seg)
                     aux_seg = (aux_seg > 0.5).float()
                     #print('aux_seg shape:',aux_seg.shape)
                     aux_seg_resize = cv2.resize(aux_seg.permute(1,2,0).cpu().detach().numpy(),(crop_aux.shape[2],crop_aux.shape[1]))
                     aux_seg_resize = torch.tensor(aux_seg_resize).unsqueeze(2)
                     #print('aux_seg shape:',aux_seg_resize.shape)
                     img_seg+=aux_seg_resize*255
                     #print('seg labels shape:',img_seg_labels.shape)
                     #print('labels shape:',labels.shape)
                     img_seg_labels[ymin:ymax,xmin:xmax] = labels*255
                   if pred_masks is not None:
                     #print('mask shape:',pred_masks.shape)
                     aux_pred_masks = pred_masks[0][i].int().to('cpu')
                     #print('aux pred_masks:',aux_pred_masks)
                     #print('pred_masks shape:',aux_pred_masks.shape)
                     #print('pred_masks:',pred_masks)
                     #print('pred_masks:',pred_masks.shape)
                     img_seg[aux_pred_masks[:,1],aux_pred_masks[:,0]] = 1
                     img_seg *= 255

            if segs is not None:
              img_seg = img_seg.cpu().detach().numpy()
              #print("img_seg shape:",img_seg.shape)
              cv2.imwrite(os.path.join(args.output_dir,folder,img_name)+"_seg.jpg",img_seg)
              img_seg_labels = img_seg_labels.unsqueeze(2)
              img_seg_labels = img_seg_labels.cpu().detach().numpy()
              cv2.imwrite(os.path.join(args.output_dir,folder,img_name)+"_seg_label.jpg",img_seg_labels)
            #print("cantidad de cajas de deteccion:",count_boxes)

            if pred_masks is not None:
              img_seg = img_seg.cpu().detach().numpy()
              #print("img_seg shape:",img_seg.shape)
              cv2.imwrite(os.path.join(args.output_dir,folder,img_name)+"_seg.jpg",img_seg)

            count_boxes = 0
            
            if output_track is not None:
               print('ids:',ids)
               if isinstance(ids, list):
                ids = torch.tensor(ids)
               ids = ids.cpu().detach().numpy()
               #print("all ids:",ids)
               #print("all delete:",delete)
               valid_ids = []
               labels, boxes, scores = output_track
               print('all_boxes:',boxes)
               print("all scores:",scores)
               #print('len boxes:',len(boxes))
               for i, box in enumerate(boxes[0]):
                #print('i track:',i)
                box = box.cpu().detach().numpy()
                scr = scores[0][i].cpu().detach().numpy()
                print('scr track:',scr)
                if ids.shape[0] == 0:
                  continue
                if scr >= thrh_track:
                   #print('id:',str(ids[i]))
                   #print("score track:",scr)
                   print('box track que paso el score:',box)
                   draw.rectangle(list(box), outline='red',)
                   #draw.text((box[0], box[1]-20), text=str(scr), fill='blue', )
                   draw.text((box[0], box[1]-30), text=str(ids[i]), fill='blue', )
                   valid_ids.append(str(ids[i]))
                   count_boxes+=1
                   #predictions.append([str(j+1),str(ids[i]),str(box[0]),str(box[1]),str(box[2]-box[0]),str(box[3]-box[1]),str(1),str(-1),str(-1),str(-1)])
                   predictions.append([str(j+1),str(ids[i]+1),str(int(box[0])),str(int(box[1])),str(int(box[2])-int(box[0])),str(int(box[3])-int(box[1])),str(1),str(-1),str(-1),str(-1)])
               #print("valid ids:",valid_ids)

            if aux_outputs_track is not None:
              labels, boxes, scores = aux_outputs_track
              #print('re_boxes despues:',boxes)
              #print('re_scores despues:',scores)
              for i, box in enumerate(boxes[0]):
                box = box.cpu().detach().numpy()
                scr = scores[0][i].cpu().detach().numpy()
                if scr >= thrh_track:
                   #print("score track:",scr)
                   draw.rectangle(list(box), outline='yellow',)
               #print("cantidad de cajas de track:",count_boxes)
            save_path = Path(os.path.join(args.output_dir,folder)) / img_name
            im.save(save_path)
        
            #aquii
            """
            for i, box in enumerate(boxes[0]):
                box = box.cpu().detach().numpy()
                scr = scores[0][i]
                if scr >= thrh:
                   draw.rectangle(list(box), outline='red',)
                   draw.text((box[0], box[1]), text=str(scr), fill='blue', )
            save_path = Path(os.path.join(args.output_dir,folder)) / img_name
            im.save(save_path)
            """
            """
            print(f"推理耗时：{time.time() - start:.4f}s")
            current = time.time() - start
            if max_time < current:
               max_time = current
            """
            #labels, boxes, scores = output
            #print('boxes:',boxes)
        """
        #print("boxes:",boxes)
        #print("scores:",scores)
            im = reader.pil_img
            im = im.resize((old_size[0], old_size[1]))
            draw = ImageDraw.Draw(im)
            for i in range(img.shape[0]):
                scr = scores[i]
                lab = labels[i][scr > thrh]
                box = boxes[i][scr > thrh]
                lab = lab.cpu().detach().numpy()
                lab = lab[:,None]
                box = box.cpu().detach().numpy()
                #print('scr:',scr)
                #print('box:',box)
                for b in box:
                    print("pasaa")
                    predictions.append([str(j),str(1),str(b[0]),str(b[1]),str(b[2]),str(b[3]),str(1),str(-1),str(-1),str(-1)])
                    #print('pred:',predictions[-1])
                    draw.rectangle(list(b), outline='red', )
                    draw.text((b[0], b[1]), text=str(lab[i]), fill='blue', )
            save_path = Path(os.path.join(args.output_dir,folder)) / img_name
            im.save(save_path)
        with open(folder+'.txt','w') as f:
             for pred in predictions:
                 f.write(','.join(pred)+'\n')
          """
        with open(folder+'.txt','w') as f:
             print('folder:',folder)
             for pred in predictions:
                 print('pred:',pred)
                 f.write(','.join(pred)+'\n')

    np.save("preds.npy", pred_dicts)

if __name__ == "__main__":
    main(get_argparser().parse_args())
