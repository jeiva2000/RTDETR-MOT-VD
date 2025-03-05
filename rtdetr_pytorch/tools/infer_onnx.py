import onnxruntime as ort 
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
import cv2
import copy
import numpy as np
import time

#file_name = '../model.onnx'
file_name = 'rtdetr_101_without_800_500.onnx'
original_im = Image.open('/home/serperzar/RT-DETR/rtdetr_pytorch/evaluation/images051/5.jpg').convert('RGB')
original_size = original_im.size

#resize image for tests
#original_im = original_im.resize((1200,2000)) # pruebaa redimencionando
"""
result = Image.new(original_im.mode, (800, 600), (0,0,0))
result.paste(original_im)
original_im = result
original_size = original_im.size
"""
im = original_im.resize((640, 640))
im_data = ToTensor()(im)[None]

print('original_size:',original_size)
sess = ort.InferenceSession(file_name)
start_time = time.time()
output = sess.run(
                 output_names=None,
                 input_feed={'images': im_data.data.numpy(), "orig_target_sizes":[list(original_size)]})
print("--- %s seconds ---" % (time.time() - start_time))
labels, boxes, scores = output

#print('labels:',labels)
#print('boxes:',boxes)
#print('scores:',scores)

img = copy.deepcopy(original_im)
img = np.asarray(img)
count = 0
for label, box, score in zip(labels[0],boxes[0],scores[0]):
    #print('box:',box)
    #print('label:',label)
    #print('score:',score)
    if score >= 0.5:
       print('box:',box)
       print('score:',score)
       img = cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])), (255,0,255), 2)
       count+=1
img = img[...,::-1]
#cv2.imwrite('test_onnx_simplify.jpg',img)
cv2.imwrite('test_onnx.jpg',img)
print('numero de predicciones:',count)
