**For my GitHub:**

Checkpoint file path(include Model weights file) : 
'work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco/latest.pth’

Config file:
'configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'

Script to use the trained model for prediction and output: output.py

**For environment:**

My version(just for suggestion):
torch                         1.8.0+cu111
torchaudio                    0.8.0
torchvision                   0.9.0+cu111

Note: 
1.The version of mmvc-full and mmdetectionn should be compatible.
2.The version of torch, torchaudio and torchvision should be compatible.
3.The official GitHub of mmdetection: 
https://github.com/open-mmlab/mmdetection.git
4.The official GitHub installation guide for mmdetection: https://mmdetection.readthedocs.io/en/latest/get_started.html
5.The official GitHub of Swin Transformer for Object Detection based on mmdetection:
https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
6.My Swin Transformer for Object Detection use:
    Backbone	Pretrain	   Lr   
    Swin-T	  ImageNet-1K	 3x	
7.

