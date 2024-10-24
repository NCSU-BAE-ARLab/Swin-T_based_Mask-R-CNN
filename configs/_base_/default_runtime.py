checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=6,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
#load_from = r"/home/wolftech/lxiang3.lab/Desktop/yx289/swin/Swin-Transformer-Object-Detection/mask_rcnn_swin_tiny_patch4_window7.pth"
#load_from = r"/home/wolftech/lxiang3.lab/Desktop/yx289/swin/Swin-Transformer-Object-Detection/work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco/latest.pth"
load_from = r"/home/wolftech/lxiang3.lab/Desktop/yx289/swin/Swin-Transformer-Object-Detection/htc_x101_32x4d_fpn_16x1_20e_coco_20200318-de97ae01.pth"
resume_from = None
workflow = [('train', 1)]
