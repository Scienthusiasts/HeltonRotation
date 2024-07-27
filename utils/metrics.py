import numpy as np
from tabulate import tabulate
import cv2
from tqdm import tqdm
import torch
from thop import profile
from thop import clever_format

# from datasets.preprocess import Transform
# 基于DOTA_devkit进行评估
import sys
sys.path.append('./DOTA_devkit')
import dota_evaluation_task1 as dota
import ResultMerge as merge









def evalDOTAmAP(pred_txt_path:str, ann_txt_path:str, imgset_file_path:str, cat_names:list[str]):
    '''基于DOTA_devkit评估模型的mAP(基于读取已经推理完的txt文件)
        # Args:
            - pred_txt_path:    模型预测的txt文件的路径(包含占位符)(class-wise)
            - ann_txt_path:     DOTA格式txt标注文件的路径(包含占位符)(img_name-wise)
            - imgset_file_path: 对应验证集的文件名list txt文件
            - cat_names:        该数据集的类别名称(list), 和pred_txt每个txt文件名一致
        # Returns:
            - map50:            mAP@.5
            - tabulate_ap_form: 评估结果表格
    '''
    ap_form = []
    map50 = 0
    # 逐类别评估
    for cat_name in tqdm(cat_names):
        rec, prec, ap, gts, dets = dota.voc_eval(pred_txt_path.format(cat_name), ann_txt_path, imgset_file_path, cat_name, ovthresh=0.5, use_07_metric=True)
        map50 += ap
        ap_form.append([cat_name, gts, dets, '%.4f'%ap])
    # 计算mAP
    map50 = map50 / len(cat_names)
    ap_form.append(['', '', '', ''])
    ap_form.append(['mAP', '', '', '%.4f'%map50])
    tabulate_ap_form = tabulate(ap_form, headers=['catagory', 'gts', 'dets', 'AP@.5'], tablefmt="psql")
    print(tabulate_ap_form)
    return map50










def mergeSplitResult(src_dir, tgt_dir):
    '''基于DOTA_devkit将split的预测结果merge成原始未裁剪图像的预测结果
        # Args:
            - src_dir: 原始评估结果txt保存目录
            - tgt_dir: merge结果txt保存目录
        # Returns:
            - None
    '''
    merge.mergebypoly(src_dir, tgt_dir)













def computeParamFLOPs(device, model, img_size:list[int], ):
    '''使用thop分析模型的运算量和参数量
    '''
    input_x = torch.rand(1, 3, img_size[0], img_size[1]).to(device)
    flops, params = profile(model, inputs=(input_x,))
    # 将结果转换为更易于阅读的格式
    flops, params = clever_format([flops, params], '%.3f')
    print(f"FLOPs↓: {flops}, 参数量↓: {params}")

















if __name__ == '__main__':
    '''eval For DOTA-v1.0'''
    # cat_names = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #             'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
    # pred_txt_path = 'F:/DeskTop/git/CKPT/HR_ckpt/yolov5l_obb/Select_IoUsmooths1_theta_rootfocalloss_lr1e-2_sgd_ddp/2024-06-28-04-39-46_train/eval_merge/Task1_{:s}.txt'
    # # ann_txt_path = 'E:/datasets/RemoteSensing/DOTA-1.0_ss_1024/val/annfiles/{:s}.txt'
    # ann_txt_path = 'E:/datasets/RemoteSensing/DOTA-1.0_1.5/val/labelTxt-v1.0/labelTxt/{:s}.txt' # no split
    # # imgset_file_path = "E:/datasets/RemoteSensing/DOTA-1.0_ss_1024/val_img_name.txt"
    # imgset_file_path = 'DOTA_devkit/DOTA_devkit_lib/evaluation_format_example/val_img_name_no_split.txt' # no split
    # map50 = evalDOTAmAP(pred_txt_path, ann_txt_path, imgset_file_path, cat_names)




    '''merge'''
    src_dir = 'log/tmp_exp/2024-07-27-11-35-09_val/eval_tmp'
    tgt_dir = 'log/tmp_exp/2024-07-27-11-35-09_val/eval_merge'
    mergeSplitResult(src_dir, tgt_dir)