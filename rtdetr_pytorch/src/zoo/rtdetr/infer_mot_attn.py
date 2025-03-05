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
        outputs_det, track_queries = self.model.forward_inference_2(images,track_queries,ind_track=True)
        attn = output_det['attn']
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
          pred_masks *=orig_target_sizes
        else:
          pred_masks = None

        outputs_det = self.postprocessor(outputs_det, orig_target_sizes)

        return outputs_det, attn

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/serperzar/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r101vd_6x_mot_coco_v3.yml", help="配置文件路径")
    parser.add_argument("--ckpt", default="../rtdetr_101vd_6x_mot_track_v3_prueba_revision/best_checkpoint.pth", help="权重文件路径")
    parser.add_argument("--video_path", default="/data/backup/serperzar/mot/dataset_mot_damages_v5/test", help="待推理图片路径")
    parser.add_argument("--output_dir", default="/home/serperzar/RT-DETR/rtdetr_pytorch/videos_output", help="输出文件保存路径")
    parser.add_argument("--device", default="cuda")

    return parser


def main(args):
    folders = os.listdir(args.video_path)
    device = torch.device(args.device)
    reader = ImageReader(resize=(640,640))
    thrh = 0.50
    thrh_track = 0.50
    if os.path.exists(args.output_dir):
       shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)
    max_time = 0
    pred_dicts = {}
    for folder in folders:
        track_queries = {}
        model = Model(confg=args.config, ckpt=args.ckpt)
        model.to(device=device)
        model.eval()
        predictions = []
        os.mkdir(os.path.join(args.output_dir,folder))
        images = os.listdir(os.path.join(args.video_path,folder))
        images = [name for name in images if name.endswith("PNG")]
        images.sort()
        pred_dicts[folder]={}
        for j, img_name in enumerate(images):
            
            old_size, img = reader(os.path.join(args.video_path,folder,img_name))
            img = img.to(device)
            size = torch.tensor([[old_size[0], old_size[1]]]).to(device)
            output_det, attn = model(img, track_queries, size)

            print('attn:',attn)

            """
            save_path = Path(os.path.join(args.output_dir,folder)) / img_name
            im.save(save_path)
            """

if __name__ == "__main__":
    main(get_argparser().parse_args())
