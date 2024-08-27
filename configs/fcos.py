import os

# train train_ddp eval test export 
MODE = 'test'
# mobilenetv3_large_100.ra_in1k  resnet50.a1_in1k  darknetaa53.c2ns_in1k cspdarknet53.ra_in1k cspresnext50.ra_in1k
FROZEBACKBONE = True
BACKBONE = 'resnet50.a1_in1k'
BACKBONE_CKPT = "F:/Desktop/git/CKPT/HD_ckpt/ckpt/backbone_resnet50.a1_in1k.pt"
LOADCKPT = f"../CKPT/HD_ckpt/yolov5s/COCO2017/bs16_lr1e-3_mosaic0.5_dropblock0.5/unfreeze/best_AP50.pt"
TESTCKPT = f"../CKPT/HD_ckpt/yolov5s/COCO2017/bs16_lr1e-3_mosaic0.5_dropblock0.5/unfreeze/best_AP50.pt"
RESUME = False
TTA = [[640,640], [832,832], [960,960]]
TTAOPEN = False

onnx_export_dir = os.path.join('onnx_ckpt', TESTCKPT.split('/')[1])
onnx_export_name = f"{TESTCKPT.split('/')[-2]}.onnx"

LOADCKPT = 'best_AP50.pt'
TESTCKPT = 'best_AP50.pt'




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
train_img_dir = "F:/Desktop/研究生/datasets/RemoteSensing/DOTA-1.0_ss_1024/train/images"
train_ann_dir = f"F:/Desktop/研究生/datasets/RemoteSensing/DOTA-1.0_ss_1024/train/{ann_name}"
# 要推理test测试集时只需修改val_img_dir:
val_img_dir = "F:/Desktop/研究生/datasets/RemoteSensing/DOTA-1.0_ss_1024/val/images"
val_ann_dir = f"F:/Desktop/研究生/datasets/RemoteSensing/DOTA-1.0_ss_1024/val/{ann_name}"
# 这两个评估时会用到, 其中eval_ann_dir里的txt是基于DOTA八参格式
imgset_file_path = "F:/Desktop/研究生/datasets/RemoteSensing/DOTA-1.0_ss_1024/val_img_name.txt"
eval_ann_dir = 'F:/Desktop/研究生/datasets/RemoteSensing/DOTA-1.0_ss_1024/val/annfiles'









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
    inferring = True,
    ckpt_path = TESTCKPT,
    T = 0.01,        
)

test = dict(
    # image image_onnx video video_onnx
    mode = 'image',
    # '''DOTA'''
    # "F:/Desktop/研究生/datasets/RemoteSensing/DOTA-1.0_ss_1024/val/images/P0019__1024__4608___0.png" 
    # P0027__1024__1322___512.png P0168__1024__1024___512.png P0262__1024__512___0.png P0476__1024__122___205.png 
    # P0660__1024__136___0.png P0833__1024__617___0.png P0086__1024__0___0.png 
    # 角度周期性问题：
    # "F:/Desktop/研究生/datasets/RemoteSensing/DIOR/JPEGImages-trainval/00268.jpg" 417
    # "F:/Desktop/研究生/datasets/RemoteSensing/DOTA-1.0_ss_1024/test/images/P0006__1024__0___505.png" P0016__1024__0___0.png P0006__1024__30___505.png
    img_path = r"F:/Desktop/研究生/datasets/RemoteSensing/DOTA-1.0_ss_1024/test/images/P0006__1024__0___505.png",
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