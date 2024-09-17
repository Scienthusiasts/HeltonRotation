import os

# train train_ddp eval test export 
MODE = 'eval'
# mobilenetv3_large_100.ra_in1k  resnet50.a1_in1k  darknetaa53.c2ns_in1k cspdarknet53.ra_in1k cspresnext50.ra_in1k
FROZEBACKBONE = True
BACKBONE = 'resnet50.a1_in1k'
BACKBONE_CKPT = "F:/Desktop/git/CKPT/HD_ckpt/ckpt/backbone_resnet50.a1_in1k.pt"
LOADCKPT = "F:/Desktop/git/CKPT/HR_ckpt/rotated_fcos/theta-weight1_adamw_lr1e-3_rotatediouloss_sample-by-freq/2024-09-07-11-44-16_train/best_AP50.pt"
TESTCKPT = "F:/Desktop/git/CKPT/HR_ckpt/rotated_fcos/theta-weight1_adamw_lr1e-3_rotatediouloss_sample-by-freq/2024-09-07-11-44-16_train/best_AP50.pt"
RESUME = False
TTA = [[640,640], [832,832], [960,960]]
TTAOPEN = False

onnx_export_dir = os.path.join('onnx_ckpt', TESTCKPT.split('/')[1])
onnx_export_name = f"{TESTCKPT.split('/')[-2]}.onnx"
# best_AP50.pt last.pt
# LOADCKPT = r"best_AP50.pt"
# TESTCKPT = r"best_AP50.pt"




'''DOTA'''
CATNUMS = 15
IMGSIZE = [1024, 1024]
ann_mode = 'yolo'
theta_mode = '-180'
cat_names = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle',
             'ship', 'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 
             'harbor', 'swimming-pool', 'helicopter']
cat_names2id = {
    'plane':0, 'baseball-diamond':1, 'bridge':2, 'ground-track-field':3,
    'small-vehicle':4, 'large-vehicle':5, 'ship':6, 'tennis-court':7,
    'basketball-court':8, 'storage-tank':9, 'soccer-ball-field':10, 
    'roundabout':11, 'harbor':12, 'swimming-pool':13, 'helicopter':14
}
reverse_map = None
ann_name = {'dota':'annfiles', 'yolo':'yolo_longside_format_annfiles'}[ann_mode]
train_img_dir = "F:/Desktop/master/datasets/RemoteSensing/DOTA-1.0_ss_size-1024_gap-200/train/images"
train_ann_dir = f"F:/Desktop/master/datasets/RemoteSensing/DOTA-1.0_ss_size-1024_gap-200/train/{ann_name}"
# 要推理test测试集时只需修改val_img_dir:
val_img_dir = "F:/Desktop/master/datasets/RemoteSensing/DOTA-1.0_ss_size-1024_gap-200/val/images"
val_ann_dir = f"F:/Desktop/master/datasets/RemoteSensing/DOTA-1.0_ss_size-1024_gap-200/val/{ann_name}"
# 这两个评估时会用到, 其中eval_ann_dir里的txt是基于DOTA八参格式
imgset_file_path = "F:/Desktop/master/datasets/RemoteSensing/DOTA-1.0_ss_size-1024_gap-200/val_img_name.txt"
eval_ann_dir = 'F:/Desktop/master/datasets/RemoteSensing/DOTA-1.0_ss_size-1024_gap-200/val/annfiles'









runner = dict(
    seed = 22,
    mode = MODE,
    resume = RESUME,
    img_size = IMGSIZE,
    epoch = 12*4,
    log_dir = './log/tmp_exp',
    log_interval = 1,
    eval_interval = 1,
    reverse_map = reverse_map,
    class_names = cat_names, 
    merge = False,
    dataset = dict(
        bs = 8,
        num_workers = 0,
        # 自定义的Dataset:
        my_dataset = dict(
            path = 'datasets/FCOSDataset.py',
            imgset_file_path = imgset_file_path,
            eval_ann_dir = eval_ann_dir,
            train_dataset = dict( 
                cat_names2id = cat_names2id,
                num_classes = CATNUMS,
                ann_dir = train_ann_dir, 
                img_dir = train_img_dir,
                input_shape = IMGSIZE, 
                img_shape = IMGSIZE,
                ann_mode = ann_mode,
                theta_mode = theta_mode,
                trainMode=True, 
                filter_empty_gt=True,
                # sample_by_freq 目前还是只支持yolo格式处理
                sample_by_freq = True,
            ),
            val_dataset = dict( 
                cat_names2id = cat_names2id,
                num_classes = CATNUMS,
                ann_dir = val_ann_dir, 
                img_dir = val_img_dir,
                input_shape = IMGSIZE, 
                img_shape = IMGSIZE,
                ann_mode = ann_mode,
                theta_mode = theta_mode,
                trainMode=False,  
                filter_empty_gt=False,  
                # sample_by_freq 目前还是只支持yolo格式处理 
                sample_by_freq = False,            
            ),
        ),
    ),

    model = dict(
        path = 'models/FCOS/FCOS.py',
        img_size = IMGSIZE, 
        num_classes = CATNUMS, 
        loadckpt = LOADCKPT,           
        backbone_name = BACKBONE,
        tta_img_size = TTA,
        backbone = dict(
            modelType = BACKBONE, 
            loadckpt = BACKBONE_CKPT, 
            pretrain = False, 
            froze = FROZEBACKBONE,
        ),
        head = dict(
            num_classes = CATNUMS,
            in_channel = 256,
            # RotatedIoULoss IoUSmoothL1Loss
            angle_loss_type = 'RotatedIoULoss',
        )
    ),
    test = dict(
        # 是否半精度推理
        half = False,
        tta = TTAOPEN,
    ),
    optimizer = dict(
        optim_type = 'adamw',
        lr = 1e-3,
        lr_min_ratio = 0.1,
        warmup_lr_init_ratio = 0.01,
    ),
)

eval = dict(
    eval_ann_dir = 'F:/Desktop/master/datasets/RemoteSensing/DOTA-1.0_1.5/val/labelTxt-v1.0/labelTxt',
    imgset_file_path = 'F:/Desktop/master/datasets/RemoteSensing/DOTA-1.0_ss_size-1024_gap-200/val_merge_img_name.txt',
    inferring = True,
    merge = True,
    ckpt_path = TESTCKPT,
    T = 0.01,        
)

test = dict(
    # image image_onnx video video_onnx
    mode = 'image',
    # '''DOTA'''

    img_path = r"samples/dota1.0/P0686__1024__0___65.png",
    save_vis_path = './samples/res1.jpg',
    # video
    # img_path = "./samples/videos/cars_people.mp4",
    # save_vis_path = './samples/videos/res1.mp4',
    ckpt_path = TESTCKPT,
    T = 0.25,
    agnostic = False,
    show_text = False,
    vis_heatmap = True,
    # onnx 权重路径
    onnx_path = os.path.join(onnx_export_dir, onnx_export_name),
)

export = dict(
    export_dir = onnx_export_dir,
    export_name = onnx_export_name,
    ckpt_path = TESTCKPT,
    export_param = dict(
        # 输入 Tensor 的名称, 如果不指定，会使用默认名字
        input_names=['input'],   
        # 输出 Tensor 的名称, 如果不指定，会使用默认名字
        output_names=['p3_head', 'p4_head', 'p5_head'],  
        # 动态输入输出设置:
        dynamic_axes = {
            # 哪个维度动态字典里索引就设置在哪个维度:
            'input':   {0: 'batch_size', 2:'input_w', 3:'input_h'},
            'p3_head': {0: 'batch_size'},
            'p4_head': {0: 'batch_size'},
            'p5_head': {0: 'batch_size'},
        }
    )
)



