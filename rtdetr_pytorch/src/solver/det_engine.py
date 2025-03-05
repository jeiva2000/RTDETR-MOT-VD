"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
from typing import Iterable

import torch
import torch.amp 

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)

from pycocotools.coco import COCO

import cv2

import numpy as np

#TrackEval
from multiprocessing import freeze_support
import TrackEval.trackeval as trackeval  # noqa: E402
import shutil
import torchvision
from torchvision.transforms import transforms
from PIL import Image

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #print("targets:",targets)
        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                        data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    print("Entra MOT")
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)
    print("scaler:",scaler)
    os.makedirs("vis_temp",exist_ok=True)
    count = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        #attn
        #model.decoder._reset_embeddings() #attn
        #samples = samples.to(device)
        samples = [sample.to(device) for sample in samples]
        #print("len samples:",len(samples))
        if isinstance(targets[0],list):
           targets=targets[0]
        #print("targets engine:",targets)
        targets = [{k: v.to(device).squeeze(0) for k, v in t.items()} for t in targets]
        #print("len targets:",len(targets))
        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets, epoch)
        else:
            track_queries = {}
            for i,(fx,target) in enumerate(zip(samples,targets)):
                
                aux_fx = fx.clone()
                aux_img = aux_fx.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255
                aux_img = cv2.cvtColor(aux_img, cv2.COLOR_BGR2RGB)
                #print("aux_img shape:",aux_img.shape)
                
                outputs, track_queries = model(fx,track_queries,target,epoch,n_frame=i)
                losses = sum(outputs.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                track_queries = {k:v.detach().clone() if torch.is_tensor(v) else v for k,v in track_queries.items()}
                #print("boxes:",track_queries["boxes"][0])
                """
                if "boxes" in track_queries:
                    width = 848
                    height = 480
                    for j, box in enumerate(track_queries["boxes"][0]):
                        box = torchvision.ops.box_convert(box, in_fmt='cxcywh', out_fmt='xyxy')
                        box = box.detach().cpu().numpy()
                        #print("box:",box)
                        cv2.rectangle(aux_img,(int(box[0]*width),int(box[1]*height)),(int(box[2]*width),int(box[3]*height)),(255,0,0),2)
                        if "ids" in track_queries:
                           id = track_queries["ids"][j].item()
                        else:
                           id = j
                        aux_img = cv2.putText(aux_img, str(id), (int(box[0]*width),int(box[1]*height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                    cv2.imwrite("vis_temp/prueba_{i}.jpg".format(i=str(i)),aux_img)
                """
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                # ema 
                if ema is not None:
                    ema.update(model)

                loss_dict_reduced = reduce_dict(outputs)
                loss_value = sum(loss_dict_reduced.values())

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

                metric_logger.update(loss=loss_value, **loss_dict_reduced)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                
        #break
        #count+=1
        #if count>100:
        #   break
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()} 

@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        outputs = model(samples)

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)
        # results = postprocessors(outputs, targets)

        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    
    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator

class ImageReader:
    def __init__(self, resize=(224,224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
             transforms.Resize((resize, resize)) if isinstance(resize, int) else transforms.Resize(
                 (resize[0], resize[1])),
            transforms.ToTensor(),
        ])
        self.resize = resize
        self.pil_img = None   #保存最近一次读取的图片的pil对象

    def __call__(self, image_path, *args, **kwargs):
        """
        读取图片
        """
        self.pil_img = Image.open(image_path).convert('RGB')
        old_size = self.pil_img.size
        return old_size, self.transform(self.pil_img).unsqueeze(0)


@torch.no_grad()
def evaluate_mot(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir, epoch):
    reader = ImageReader(resize=(640,640))
    #root_path = '/data/backup/serperzar/mot/dancetrack/aux_mot_dataset/train'
    #coco = COCO('/data/backup/serperzar/mot/dataset_dancetrack_coco_format/annotations/instances_train.json')
    #folders = os.listdir('/data/backup/serperzar/mot/dancetrack/aux_mot_dataset/train')
    #coco = COCO('/data/backup/serperzar/mot/dataset_mot_coco_format_v5/annotations/instances_test.json')
    root_path = '/data/backup/serperzar/mot/dancetrack/mot_dataset/val' #estar cambiando
    coco = COCO('/data/backup/serperzar/mot/aux_dancetrack/annotations/instances_val.json')
    folders = os.listdir(root_path)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    os.makedirs('TrackEval/data/trackers/mot_challenge/Dancetrack-val/rtdetr_mot_dance',exist_ok=True)
    os.makedirs('TrackEval/data/trackers/mot_challenge/Dancetrack-val/rtdetr_mot_dance/data',exist_ok=True)
    for folder in folders:
        track_queries = {}
        model.eval()
        criterion.eval()
        images = os.listdir(os.path.join(root_path,folder))
        images = [name for name in images if name.endswith(("PNG","jpg"))]
        images.sort()
        #print('len images:',len(images))
        list_track = []
        list_ids_track = []
        #save_file = os.path.join('TrackEval/data/trackers/mot_challenge/MOTD-test/rtdetr_mot/data/',folder+'.txt')
        save_file = os.path.join('TrackEval/data/trackers/mot_challenge/Dancetrack-val/rtdetr_mot_dance/data/',folder+'.txt')
        for j, img_name in enumerate(images):
            old_size, img = reader(os.path.join(root_path,folder,img_name))
            old_size = torch.tensor([[old_size[0], old_size[1]]]).to(device)
            img = img.to(device)
            aux_det_outputs, track_queries = model.forward_inference_2(img,track_queries,n_frame=j)
            #print("track_queries:",track_queries)
            if 'pred_logits_2' not in track_queries or track_queries['pred_logits_2'] is None:
                track_queries['pred_logits_2'] = torch.tensor([[0,0]]).unsqueeze(0).to(device)
                track_queries['pred_boxes_2'] = torch.tensor([[0,0,0,0]]).unsqueeze(0).to(device)
            track_queries['pred_logits'] = track_queries['pred_logits_2']
            track_queries['pred_boxes'] = track_queries['pred_boxes_2']
            if 'boxes' in track_queries:
                track_queries['pred_logits'] = track_queries['logits']
                track_queries['pred_boxes'] = track_queries['boxes']
            if 'ids' in track_queries:
                ids_track = track_queries['ids']
            else:
                ids_track = None
            postprocessors.num_top_queries = track_queries['pred_boxes'].shape[1]
            results_track = postprocessors(track_queries, old_size,ind_track=True)[0]
            postprocessors.num_top_queries = aux_det_outputs['pred_boxes'].shape[1]
            results_det = postprocessors(aux_det_outputs, old_size)[0]
            list_track.append(results_track)
            list_ids_track.append(ids_track)

        #print("len list_track:",len(list_track))
        #print("len list ids:",len(list_ids_track))

        with open(save_file,'w') as f:
            for i_x,(results_track,ids_track) in enumerate(zip(list_track,list_ids_track)):
                if ids_track is None:
                    #print("no pasaa")
                    continue
                for box,score,aux_ids in zip(results_track['boxes'],results_track['scores'],ids_track):
                    aux_ids = aux_ids.cpu().numpy()
                    box = box.cpu().numpy()
                    #print('aux_boxes:',box)
                    scr = score.cpu().detach().numpy()
                    if scr >= 0.3:
                        f.write(str(i_x+1)+','+str(aux_ids+1)+','+str(int(box[0]))+','+str(int(box[1]))+','+str(int(box[2])-int(box[0]))+','+str(int(box[3])-int(box[1]))+',1,-1,-1,-1'+'\n')
        if not os.path.exists(save_file):
            open(save_file, 'w').close()
    tracker = 'rtdetr_mot_dance' #'rtdetr_mot'
    dataset_eval = 'Dancetrack' #'MOTD'
    freeze_support()
    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    #default_eval_config['TRACKERS_TO_EVAL'] = ['rtdetr_mot']
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_dataset_config['BENCHMARK'] = dataset_eval #'MOTD'
    default_dataset_config['TRACKERS_TO_EVAL'] = [tracker] #['rtdetr_mot']
    default_metrics_config = {'METRICS': ['CLEAR', 'Identity','HOTA'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    #eval_config['SPLIT_TO_EVAL'] = 'test' #solo para test
    dataset_config['SPLIT_TO_EVAL'] = 'val' #'test' #solo para test
    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    #print(output_res['MotChallenge2DBox']['rtdetr_mot'].keys())
    #print('output_res:',output_res['MotChallenge2DBox']['rtdetr_mot']['details'][0]['COMBINED_SEQ'])
    #print('output_msg:',output_msg)
    stats = {'HOTA_AUC':[output_res['MotChallenge2DBox'][tracker]['details'][0]['COMBINED_SEQ']['HOTA___AUC']]}
    return stats, None
"""
@torch.no_grad()
def evaluate_mot(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir, epoch):
    #coco = COCO('/data/backup/serperzar/mot/dataset_mot_coco_format_v5/annotations/instances_val.json')
    #coco = COCO('/data/backup/serperzar/mot/dataset_mot_coco_format_v5/annotations/instances_test.json')
    #folders = os.listdir('/data/backup/serperzar/mot/dataset_mot_damages_v5/test')
    coco = COCO('/data/backup/serperzar/mot/dataset_dancetrack_coco_format/annotations/instances_val.json')
    folders = os.listdir('/data/backup/serperzar/mot/dancetrack/mot_dataset/val')
    model.eval()
    criterion.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    dict_preds = {}
    for folder_name in folders:
        dict_preds[folder_name] = {}
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        #print('targets:',targets)
        samples = [sample.to(device) for sample in samples]
        if isinstance(targets[0],list):
           targets=targets[0]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        det_outputs = []
        outputs = []
        for i in range(len(targets)):
            info = coco.loadImgs(targets[i]['image_id'].item())
            folder_name = info[0]['file_name'].split('/')[0]
            img_name = info[0]['file_name'].split('/')[1]
            if folder_name not in dict_preds:
                dict_preds[folder_name] = {}
            #print('img:',samples[i])
            dict_preds[folder_name][img_name] = {'img':samples[i],'orig_target_size':targets[i]['orig_size']}

    save_folder = 'TrackEval/data/trackers/mot_challenge/MOTD-test/rtdetr_mot'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        os.mkdir(save_folder+'/data')

    for k,v in dict_preds.items():
        save_file = os.path.join('TrackEval/data/trackers/mot_challenge/MOTD-test/rtdetr_mot/data/',k+'.txt')
        list_track = []
        list_ids_track = []
        track_queries = {}
        #if k == '5':
        #    print('k:',k)
        for i_x,(k0,v0) in enumerate(v.items()):
            #if k == '5':
                #print('img_name:',k0)
                #print('shape:',v0['img'].shape)
            #print('imagen a enviar a inference2:',v0['img'])
            #print('img_name:',k0)
            aux_det_outputs, track_queries = model.forward_inference_2(v0['img'],track_queries)
            if 'pred_logits_2' not in track_queries or track_queries['pred_logits_2'] is None:
                track_queries['pred_logits_2'] = torch.tensor([[0,0]]).unsqueeze(0).to(device)
                track_queries['pred_boxes_2'] = torch.tensor([[0,0,0,0]]).unsqueeze(0).to(device)
            track_queries['pred_logits'] = track_queries['pred_logits_2']
            track_queries['pred_boxes'] = track_queries['pred_boxes_2']
            if 'boxes' in track_queries:
                track_queries['pred_logits'] = track_queries['logits']
                track_queries['pred_boxes'] = track_queries['boxes']
            if 'ids' in track_queries:
                ids_track = track_queries['ids']
            else:
                ids_track = None
            postprocessors.num_top_queries = track_queries['pred_boxes'].shape[1]
            #print('orig_target_size:',v0['orig_target_size'])
            #if k == '5':
            #    print('results_track before process:',track_queries['pred_boxes'])
            results_track = postprocessors(track_queries, v0['orig_target_size'],ind_track=True)[0]
            #if k == '5':
            #    print('results_track post:',results_track)
            postprocessors.num_top_queries = aux_det_outputs['pred_boxes'].shape[1]
            results_det = postprocessors(aux_det_outputs, v0['orig_target_size'])[0]
            list_track.append(results_track)
            list_ids_track.append(ids_track)
            #print('results_track:',results_track)
            #print('ids_track:',ids_track)
            #print('save file a guardar:',save_file)
        with open(save_file,'w') as f:
            for i_x,(results_track,ids_track) in enumerate(zip(list_track,list_ids_track)):
                if ids_track is None:
                    continue
                for box,score,aux_ids in zip(results_track['boxes'],results_track['scores'],ids_track):
                    aux_ids = aux_ids.cpu().numpy()
                    box = box.cpu().numpy()
                    #print('aux_boxes:',box)
                    scr = score.cpu().detach().numpy()
                    if scr >= 0.3:
                        f.write(str(i_x+1)+','+str(aux_ids+1)+','+str(int(box[0]))+','+str(int(box[1]))+','+str(int(box[2])-int(box[0]))+','+str(int(box[3])-int(box[1]))+',1,-1,-1,-1'+'\n')
        if not os.path.exists(save_file):
            #print('no existe')
            open(save_file, 'w').close()
            #with open(save_file,'w') as f:
            #    f.write('1,0,0,0,0,0,1,-1,-1,-1\n')

    freeze_support()
    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    #default_eval_config['TRACKERS_TO_EVAL'] = ['rtdetr_mot']
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_dataset_config['BENCHMARK'] = 'MOTD'
    default_dataset_config['TRACKERS_TO_EVAL'] = ['rtdetr_mot']
    default_metrics_config = {'METRICS': ['CLEAR', 'Identity','HOTA'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    #eval_config['SPLIT_TO_EVAL'] = 'test' #solo para test
    dataset_config['SPLIT_TO_EVAL'] = 'test' #solo para test
    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    #print(output_res['MotChallenge2DBox']['rtdetr_mot'].keys())
    #print('output_res:',output_res['MotChallenge2DBox']['rtdetr_mot']['details'][0]['COMBINED_SEQ'])
    #print('output_msg:',output_msg)
    stats = {'HOTA_AUC':[output_res['MotChallenge2DBox']['rtdetr_mot']['details'][0]['COMBINED_SEQ']['HOTA___AUC']]}
    return stats, None
"""
"""   
@torch.no_grad()
def evaluate_mot(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir, epoch):
    coco = COCO('/data/backup/serperzar/mot/dataset_mot_coco_format_v5/annotations/instances_val.json')
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None

    os.makedirs('vis_eval',exist_ok=True)
    cont_iter = 0
    colors = np.random.randint(255,size=(300,3))
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        os.makedirs('vis_eval/'+str(cont_iter),exist_ok=True)
        samples = [sample.to(device) for sample in samples]
        if isinstance(targets[0],list):
           targets=targets[0]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        det_outputs = []
        outputs = []
        track_queries = {}
        for sample in samples:
            aux_det_outputs, track_queries = model(sample,track_queries,epoch=epoch,return_two_outputs=True)
            det_outputs.append(aux_det_outputs)
            outputs.append(track_queries.copy())
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        for i in range(len(targets)):
            ids_track = None
            if 'pred_logits_2' not in outputs[i] or outputs[i]['pred_logits_2'] is None:
                outputs[i]['pred_logits_2'] = torch.tensor([[0,0]]).unsqueeze(0).to(device)
                outputs[i]['pred_boxes_2'] = torch.tensor([[0,0,0,0]]).unsqueeze(0).to(device)
            outputs[i]['pred_logits'] = outputs[i]['pred_logits_2']
            outputs[i]['pred_boxes'] = outputs[i]['pred_boxes_2']
            if 'ids' in outputs[i]:
                ids_track = outputs[i]['ids']
            else:
                ids_track = None
            postprocessors.num_top_queries = outputs[i]['pred_boxes'].shape[1]
            results_track = postprocessors(outputs[i], orig_target_sizes[i],ind_track=True)[0]
            postprocessors.num_top_queries = det_outputs[i]['pred_boxes'].shape[1]
            results_det = postprocessors(det_outputs[i], orig_target_sizes[i])[0]
            info = coco.loadImgs(targets[i]['image_id'].item())
            img = cv2.imread('/data/backup/serperzar/mot/dataset_mot_damages_v4/val/'+info[0]['file_name'])
            #print('file_name:',info[0]['file_name'])
            #print('outputs i:',outputs[i]['pred_boxes'])
            #targets
            for box in targets[i]['boxes'].cpu().numpy()[0]:
                aux_size = orig_target_sizes[i][0].cpu().numpy()
                scale_x = aux_size[0] / 640
                scale_y = aux_size[1] / 640
                img = cv2.rectangle(img,(int(box[0]*scale_x),int(box[1]*scale_y)),(int(box[2]*scale_x),int(box[3]*scale_y)),(255,0,0),2)
            #pred boxes
            
            for box, score in zip(results_det['boxes'].cpu().numpy(),results_det['scores'].cpu().numpy()):
                if score > 0.5:
                    img = cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2)
            
            #track boxes
            #print('boxes:',results_track['boxes'])
            #print('scores',results_track['scores'])
            #print('ids',ids_track)
            
            if results_track['boxes'] is not None and results_track['scores'] is not None and ids_track is not None:
                aux_count = 0
                #print('boxes in det engine:',results_track['boxes'])
                for box, score, ids in zip(results_track['boxes'].cpu().numpy(),results_track['scores'].cpu().numpy(),ids_track.cpu().numpy()):
                    if score > 0.1:
                        color = tuple([int(x) for x in colors[aux_count]])
                        img = cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),color,2)
                        img = cv2.putText(img, str(ids), (int(box[0])+5,int(box[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA) 
                    aux_count+=1
            
            cv2.imwrite('vis_eval/'+str(cont_iter)+'/'+str(i)+'.jpg',img)
            
            if epoch < 0:
               res = {targets[i]['image_id'].item(): results_det} #det
            else:
               res = {targets[i]['image_id'].item(): results_track} #track
            
            #res = {targets[i]['image_id'].item(): results_track} #track
            if coco_evaluator is not None:
                coco_evaluator.update(res)
        cont_iter+=1
        
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    
    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist() 
    return stats, coco_evaluator
"""
