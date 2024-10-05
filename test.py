import os
import cv2
import json
import torch
import argparse
import importlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.ops import nms
from pycocotools.coco import COCO

# 自定义模块
from utils.metrics import *
from utils.util import *
from utils.exportUtils import onnxInferenceSingleImg, onnxInferenceVideo







class Test():

    def __init__(self, path, model, img_size, class_names, device, half, tta):
        self.device = device
        self.tta = tta
        self.half = half
        self.model = model
        # 半精度推理:
        if self.half: self.model.half()
        self.img_size = img_size
        self.cat_nums = len(class_names)
        self.class_names = class_names
        # 动态导入
        self.tf = dynamic_import_class(path, 'Transform')(self.img_size)
        '''每个类别都获得一个随机颜色'''
        self.image2color = dict()
        for i, cat in enumerate(class_names):
            self.image2color[cat] = (np.random.random((1, 3)) * 0.7 + 0.3).tolist()[0]



    def predict(self, mode, img_path, onnx_path=False, save_vis_path=None, ckpt_path=None, T=0.3, agnostic=False, vis_heatmap=False, show_text=True, **kwargs):
        '''推理一张图
            Args:
                - path:          图片/视频路径
                - save_vis_path: 可视化图像/视频保存路径

            Returns:
                - boxes:       网络回归的box坐标    [obj_nums, 4]
                - box_scores:  网络预测的box置信度  [obj_nums]
                - box_classes: 网络预测的box类别    [obj_nums]
        '''
        # 调用模型自己的推理方法
        if mode == 'image':
            boxes, box_scores, box_classes = inferenceSingleImg(
                self.model, self.device, self.class_names, self.image2color, self.img_size, self.tf, img_path, save_vis_path, ckpt_path, T, agnostic, show_text, vis_heatmap, self.half, self.tta
                )
            return boxes, box_scores, box_classes
        if mode == 'image_onnx':
            boxes, box_scores, box_classes = onnxInferenceSingleImg(
                self.model, self.device, self.class_names, self.image2color, self.img_size, self.tf, img_path, onnx_path, save_vis_path, T, agnostic, show_text, vis_heatmap
                )            
            return boxes, box_scores, box_classes
        if mode == 'video':
            inferenceVideo(self.model, self.device, self.class_names, self.image2color, self.img_size, self.tf, img_path, save_vis_path, ckpt_path, T, agnostic, show_text, self.half)
        if mode == 'video_onnx':   
            onnxInferenceVideo(self.model, self.device, self.class_names, self.image2color, self.img_size, self.tf, img_path, onnx_path, save_vis_path, T, agnostic, show_text)







    def formatOneImg(self, img_name, img_path, pred_per_cat_list, T=0.3, agnostic=False, reverse_map=None):
        '''推理一张图,并返回DOTA格式下每一个字段
            Args:
                - img_path: 图片路径
                - image_id: 可视化图像保存路径
                - anns_dict: COCO格式下'annotations'字段

            Returns:
                - boxes:       网络回归的box坐标    [obj_nums, 4]
                - box_scores:  网络预测的box置信度  [obj_nums]
                - box_classes: 网络预测的box类别    [obj_nums]
        '''
        boxes, box_scores, box_classes = self.predict(mode='image', img_path=img_path, T=T, agnostic=agnostic)
        #  检测出物体才继续    
        if len(boxes) == 0: 
            return pred_per_cat_list
        # 将5参旋转框表示法转换为8参四边形表示法
        poly_boxes = rbox2PolyNP(boxes)
        # 按类别整理预测结果
        for box, score, cls in zip(poly_boxes, box_scores, box_classes):
            box = [str(b) for b in box]
            pred_per_cat_list[cls].append([img_name.split('.')[0], str(score)]+box)

        return pred_per_cat_list








    def genPredDOTAtxt(self, img_dir, imgset_file_path, val_ann_dir, pred_save_dir, T=0.01, agnostic=False, model=None, inferring=True, ckpt_path=None, reverse_map=None, fuse=False, merge=False):
        '''生成预测结果, 并将预测结果整理为DOTA_devkit评估的格式
            Args:
                - img_dir:          数据集图像文件夹
                - imgset_file_path: 对应验证集的文件名list txt文件
                - val_ann_dir:      验证集的ann txt文件根目录
                - pred_save_dir:    模型推理得到的txt文件根目录
                - T:                置信度(模型推理超参)
                - model:            导入的模型(当mode=='eval'时有效)
                - inferring:        是否让网络推理一遍数据集并生成txt
                - merge:            是否将裁剪后的预测结果拼回去评估

            Returns:
                - mAP:   所有类别平均 AP@.5:.95          
        '''
        img_dir_list = os.listdir(img_dir)
        # 当某个类别完全没有预测实例时,使用以下占位(基于DOTA1.0-split val)
        occupy_tmp = 'P0003__1024__0___0 0.035342857 465.94525 235.61427 388.495 141.69347 510.94528 40.716816 588.3955 134.63762' 
        pred_save_dir += '/eval_tmp'
        if not os.path.isdir(pred_save_dir ):os.mkdir(pred_save_dir)
        # 是否导入权重
        if ckpt_path != None:
            print('load_ckpt: ', ckpt_path)
            self.model.load_state_dict(torch.load(ckpt_path))
            # self.model = loadWeightsBySizeMatching(self.model, ckpt_path)
            # 半精度推理(貌似还有问题,推理速度和全精度差不多):
            if self.half: self.model.half()
            # yolov8:
            if fuse:
                self.model = self.model.fuse()
        self.model = model.eval()
        '''是否在线推理, 在线推理则会在线生成一个eval的txt文件夹(和DOTA提交服务器的格式一致)'''
        if inferring:
            print('inferring...')
            # 初始化pred_per_cat_list存储逐类别预测实例
            pred_per_cat_list = [[] for _ in range(self.cat_nums)]
            for img_name in tqdm(img_dir_list):
                img_path = os.path.join(img_dir, img_name)
                pred_per_cat_list = self.formatOneImg(img_name, img_path, pred_per_cat_list, T=T, agnostic=agnostic, reverse_map=reverse_map)
            # 逐类别生成预测的txt文件
            print('generate class-wise pred txt files...')
            for i, cat_name in tqdm(enumerate(self.class_names)):
                pred_txt_path = os.path.join(pred_save_dir, f'Task1_{cat_name}.txt')
                with open(pred_txt_path, 'w') as txt:
                    for inst in pred_per_cat_list[i]:
                        txt.write(' '.join(inst)+'\n')
                    # 当某个类别完全没有预测实例时,使用以下下占位(基于DOTAval)
                    if len(pred_per_cat_list[i])==0:
                        print(f'no instances in {pred_txt_path}')
                        txt.write(occupy_tmp+'\n')
        # 采用DOTA_devkit进行评估:
        if merge:
            merge_pred_save_dir = pred_save_dir.replace('eval_tmp', 'eval_tmp_merge')
            mergeSplitResult(pred_save_dir, merge_pred_save_dir)
            pred_save_dir = merge_pred_save_dir

        pred_save_path = pred_save_dir + '/Task1_{:s}.txt'
        val_ann_path = val_ann_dir + '/{:s}.txt'
        map50, mrecall, mprecision = evalDOTAmAP(pred_save_path, val_ann_path, imgset_file_path, self.class_names)
        return map50, mrecall, mprecision











def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='config file')
    args = parser.parse_args()
    return args



def import_module_by_path(module_path):
    """根据给定的完整路径动态导入模块(config.py)
    """
    spec = importlib.util.spec_from_file_location("module_name", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module













# for test only:
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = getArgs()
    # 使用动态导入的模块
    config_path = args.config
    config_file = import_module_by_path(config_path)
    # 调用动态导入的模块的函数
    config = config_file.test_config
    tester = Test(config['mode'], config['img_size'], config['class_names'], device, ckpt_path=config['ckpt_path'], colors=config['colors'])
    if config['mode']=='test':
        tester.predictOneImg(config['img_path'], config['save_res_path'], T=config['confidence'])
    if config['mode']=='eval':
        tester.genPredJsonAndEval(config['json_path'], config['img_dir'], config['pred_json_path'], T=config['confidence'], inferring=False, printLog=True)
    