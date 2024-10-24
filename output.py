import os
import mmcv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
checkpoint_file = 'work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco/latest.pth'

# choose device
device = 'cuda:0' #one GPU
model = init_detector(config_file, checkpoint_file, device=device)


input_folder = 'data/coco/test_select_jpg'  
output_folder = 'data/coco/test_select_jpg_demo'  

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):  
        img_path = os.path.join(input_folder, filename)
        img = mmcv.imread(img_path)
        result = inference_detector(model, img)
        img_show = img.copy()

        total_masks = sum([len(masks) for masks in result[1]])  # #mask
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(total_masks)]  
        color_index = 0
        bbox_color = (20, 20, 20)
        bbox_thickness = 2
        for cls_id, (bboxes, masks) in enumerate(zip(result[0], result[1])):
            for bbox, mask in zip(bboxes, masks):
                color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                cls_color = colors[color_index] 
                color_mask[mask] = cls_color
                img_show = cv2.addWeighted(img_show, 1, color_mask, 0.5, 0)

                x1, y1, x2, y2 = map(int, bbox[:4])
                score = round(bbox[4], 1)
                cv2.rectangle(img_show, (x1, y1), (x2, y2), bbox_color, bbox_thickness)
                cv2.putText(img_show, f'{score:.1f}', (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, bbox_color, 1)
                color_index += 1

        output_img_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_img_path, cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR))


