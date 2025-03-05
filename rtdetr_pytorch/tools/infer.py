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


class ImageReader:
    def __init__(self, resize=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
            # transforms.Resize((resize, resize)) if isinstance(resize, int) else transforms.Resize(
            #     (resize[0], resize[1])),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ])
        self.resize = resize
        self.pil_img = None   #保存最近一次读取的图片的pil对象

    def __call__(self, image_path, *args, **kwargs):
        """
        读取图片
        """
        self.pil_img = Image.open(image_path).convert('RGB').resize((self.resize, self.resize))
        return self.transform(self.pil_img).unsqueeze(0)




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
        self.cfg.model.load_state_dict(state)

        self.model = self.cfg.model.deploy()
        self.postprocessor = self.cfg.postprocessor.deploy()
        # print(self.postprocessor.deploy_mode)
        
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)



def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/serperzar/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml", help="配置文件路径")
    parser.add_argument("--ckpt", default="/home/serperzar/RT-DETR/rtdetr_pytorch/weights/rtdetr_r50vd_6x_coco_from_paddle.pth", help="权重文件路径")
    parser.add_argument("--image", default="/home/serperzar/RT-DETR/rtdetr_pytorch/images/34583.jpg", help="待推理图片路径")
    parser.add_argument("--output_dir", default="/home/serperzar/RT-DETR/rtdetr_pytorch/images/ouput", help="输出文件保存路径")
    parser.add_argument("--device", default="cpu")

    return parser


def main(args):
    img_path = Path(args.image)
    device = torch.device(args.device)
    reader = ImageReader(resize=640)
    model = Model(confg=args.config, ckpt=args.ckpt)
    model.to(device=device)

    img = reader(img_path).to(device)
    size = torch.tensor([[img.shape[2], img.shape[3]]]).to(device)
    start = time.time()
    output = model(img, size)
    print(f"推理耗时：{time.time() - start:.4f}s")
    labels, boxes, scores = output
    im = reader.pil_img
    draw = ImageDraw.Draw(im)
    #thrh = 0.6
    thrh = 0.5

    for i in range(img.shape[0]):

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]


        for b in box:
            draw.rectangle(list(b), outline='red', )
            draw.text((b[0], b[1]), text=str(lab[i]), fill='blue', )

    save_path = Path(args.output_dir) / img_path.name
    im.save(save_path)
    print(f"检测结果已保存至:{save_path}")

if __name__ == "__main__":
    main(get_argparser().parse_args())
