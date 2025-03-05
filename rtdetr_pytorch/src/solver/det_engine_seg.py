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
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        #samples = samples.to(device)
        samples = [sample.to(device) for sample in samples]
        #print("len samples:",len(samples))
        if isinstance(targets[0],list):
           targets=targets[0]
        #print("targets:",targets)
        targets = [{k: v.to(device).squeeze(0) for k, v in t.items()} for t in targets]
        #print("len targets:",len(targets))
        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets, epoch)
        else:
            track_queries = {}
            for i,(fx,target) in enumerate(zip(samples,targets)):
                if i == len(samples)-1:
                    outputs, track_queries = model(fx,track_queries,target,epoch,last=True)
                else:
                    outputs, track_queries = model(fx,track_queries,target,epoch)
                losses = sum(outputs.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                track_queries = {k:v.detach().clone() if torch.is_tensor(v) else v for k,v in track_queries.items()}

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


@torch.no_grad()
def evaluate_mot(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir, epoch):
    coco = COCO('/data/backup/serperzar/mot/dataset_mot_coco_format/annotations/instances_val.json')
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
        """
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        for output, det_outputs, orig_target_size in zip(outputs, det_outputs, orig_target_sizes):
            print('entraa')
            print('orig_target_size:',orig_target_size)
            #print('output:',output)
            aux_output = {}
            if 'pred_logits_2' not in output or output["pred_logits_2"] is None: #truquito
               aux_pred_logits = torch.tensor([[0,0]]).unsqueeze(0).to(device)
               aux_pred_boxes = torch.tensor([[0,0,0,0]]).unsqueeze(0).to(device)
               #print('track_queries van vacias')
            else:
               #print('track queries van llenas')
               aux_pred_logits = output['pred_logits_2']
               aux_pred_boxes = output['pred_boxes_2']
            aux_output["pred_logits"] = aux_pred_logits
            aux_output["pred_boxes"] = aux_pred_boxes
            #print('output shape:',output['pred_boxes'].shape)
            #postprocessors.num_top_queries = aux_output['pred_boxes'].shape[1]
            postprocessors.num_top_queries = det_outputs['pred_boxes'].shape[1]
            #results = postprocessors(aux_output, orig_target_size)
            #print('det_outputs:',det_outputs)
            results_det = postprocessors(det_outputs, orig_target_size)
            #print('orig_target_size:',orig_target_size)
            #print('pred_results:',results)
            print('epoch:',epoch)
            if epoch < 2:
                res = {target['image_id'].item(): output_r for target, output_r in zip(targets, results_det)}
            #else:
            #    res = {target['image_id'].item(): output_r for target, output_r in zip(targets, results)}
            #res_2 = {target['image_id'].item(): [output_r, target, output_det] for target, output_r, output_det in zip(targets, results, results_det)}
            res_2 = {target['image_id'].item(): [target, output_det, output_boxes_logits] for target, output_det, output_boxes_logits in zip(targets, results_det, det_outputs['pred_boxes'])}
            for k,v in res_2.items():
                info = coco.loadImgs(k)
                #print('file_name:',info[0]['file_name'])
                #print('/data/backup/serperzar/mot/dataset_mot_damages_v4/val/'+info[0]['file_name'])
                img = cv2.imread('/data/backup/serperzar/mot/dataset_mot_damages_v4_aux/val/'+info[0]['file_name'])
                print('file_name:',info[0]['file_name'])
                print('output_logits:',v[2])
                
                #boxes = v[0]['boxes'].cpu().numpy()
                #scores = v[0]['scores'].cpu().numpy()
                #for box,score in zip(boxes,scores):
                    #print('box r:',box)
                #    if score > 0.1:
                #       img = cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),2)
                boxes = v[1]['boxes'].cpu().numpy()
                scores = v[1]['scores'].cpu().numpy()
                for box,score in zip(boxes,scores):
                    #print('box r:',box)
                    if score > 0.1:
                       img = cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2)
                #print('vis_eval/'+info[0]['file_name'].split('/')[-1])
                for box in v[0]['boxes'].cpu().numpy()[0]:
                    #print('box_t:',box)
                    aux_size = orig_target_size[0].cpu().numpy()
                    scale_x = aux_size[0] / 640
                    scale_y = aux_size[1] / 640
                    #print('box reshape:',int(box[0]*scale_x),int(box[1]*scale_y),int(box[2]*scale_x),int(box[3]*scale_y))
                    img = cv2.rectangle(img,(int(box[0]*scale_x),int(box[1]*scale_y)),(int(box[2]*scale_x),int(box[3]*scale_y)),(255,0,0),2)
                    
                    #img = cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),2)
                cv2.imwrite('vis_eval/'+info[0]['file_name'].split('/')[-1],img)
            #print('targets:',res)
            if coco_evaluator is not None:
               coco_evaluator.update(res)
        """
    
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
