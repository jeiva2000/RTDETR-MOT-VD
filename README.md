# RTDETR-MOT-VD
RTDETR-MOT-VEHICLE-DAMAGES-DETECTION


https://github.com/user-attachments/assets/98014856-f404-4249-b6c6-9b6b20ce27c1


## Data
Download our dataset here: https://docs.google.com/forms/d/e/1FAIpQLSdweTh_4EQD97s-45X9L_wu82IdTyfUwBS3d53mFskJBxXFzg/viewform?usp=header

## Training

python tools/train.py -c configs/rtdetr/rtdetr_r101vd_6x_mot_coco_v3.yml

## Inference

python infer_mot.py --config ../configs/rtdetr/rtdetr_r101vd_6x_mot_coco_v3.yml --ckpt model.pth

Comming soon.

