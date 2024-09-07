# coding=utf-8
import numpy as np
import torch
from functools import partial
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import random
import albumentations as A
from pycocotools.coco import COCO
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.util import *
from utils.YOLOAnchorUtils import *
from utils.rotateUtils import *
from datasets.preprocess import *









class DOTA2LongSideFormatYOLODataset(Dataset):
    '''基于DOTA数据集YOLO格式的长边表示法数据集读取方式, 角度的范围在[-180, 0)
    '''

    def __init__(self, num_classes, cat_names2id, anchors, anchors_mask, ann_dir, img_dir, img_shape=[1024, 1024], input_shape=[800, 800], ann_mode='yolo', theta_mode='-180', trainMode=True, filter_empty_gt=True, sample_by_freq=False):
        '''__init__() 为默认构造函数，传入数据集类别（训练或测试），以及数据集路径

        Args:
            - annPath:     COCO annotation 文件路径
            - imgDir:      图像的根目录
            - inputShape: 网络要求输入的图像尺寸
            - trainMode:   训练集/测试集
            - trainMode:   训练集/测试集
            -  map:        categories字段映射
        Returns:
            FRCNNDataset
        '''      
        self.sample_by_freq = sample_by_freq
        self.mode = trainMode
        self.ann_mode = ann_mode
        self.theta_mode = theta_mode
        self.num_classes = num_classes
        self.cat_names2id = cat_names2id
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.img_shape = img_shape
        self.input_shape = input_shape
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.ann_list = os.listdir(ann_dir)
        if filter_empty_gt:self.filter_empty_gt()
        # 数据集大小
        self.datasetNum = len(self.ann_list)
        # 图像增强变换
        self.tf = Transform('coco', img_shape, input_shape, self.datasetNum)
        if self.sample_by_freq:
            # 统计每个类别下有哪些图片, 并给出每个类别的采样比例(类别数量少就采样概率大一些)
            self.cat_img_dict, self.sampling_ratio = count_imgs_per_cat(img_dir, ann_dir, num_classes)


    def filter_empty_gt(self):
        '''训练时过滤掉那些GT为空的文件
        '''
        filter_ann_list = []
        before_filter_img_num = len(self.ann_list)
        print('filter empty gt...')
        for ann_name in self.ann_list:
            ann_path = os.path.join(self.ann_dir, ann_name)
            # 如果文件大小为 0 字节，则文件为空
            if os.path.getsize(ann_path) != 0:
                filter_ann_list.append(ann_name)
        
        self.ann_list = filter_ann_list
        print(f'filter img total num: {before_filter_img_num-len(self.ann_list)}')

        
    def __len__(self):
        '''重载data.Dataset父类方法, 返回数据集大小
        '''
        return self.datasetNum

    def __getitem__(self, index):
        '''重载data.Dataset父类方法, 获取数据集中数据内容
        '''   
        # 通过index获得图像, 图像的框, 以及框的标签('dota'是获取dota标注格式的8个坐标, 而yolo是获得yolo格式的五参表示法(cls, cx, cy, w, h, theta))
        if self.ann_mode == 'dota':
            image, boxes, angle, labels = self.getDataByIndex(index)
        elif self.ann_mode == 'yolo':
            image, boxes, angle, labels = self.getDataByIndexYOLOLongSide(index)
        # 数据预处理与增强
        image, boxes, angle, labels = self.augment(image, boxes, angle, labels)

        keep = []
        '''下面的循环是为了剔除掉超出图像边界太多的旋转框'''
        for id, (instBox, a) in enumerate(zip(boxes, angle)):
            rect = longsideFormat2OpenCVFormat(instBox[0], instBox[1], instBox[2], instBox[3], a)
            poly = np.float32(cv2.boxPoints(rect))
            # 每个框如果超出图像边界的角点个数在两个及以上, 则舍弃
            cx_exceed_num = sum((poly[:, 0]>self.img_shape[0]) + (poly[:, 0]<0))
            cy_exceed_num = sum((poly[:, 1]>self.img_shape[1]) + (poly[:, 1]<0))
            if cx_exceed_num>2 or cy_exceed_num>2:
                continue
            keep.append(id)
        # cxcywh -> norm(cxcywh)
        boxes[:, [0, 2]] /= self.img_shape[1]
        boxes[:, [1, 3]] /= self.img_shape[0]
        boxes = np.concatenate((boxes, angle.reshape(-1, 1), labels.reshape(-1, 1)), axis=1)[keep]
        # len(y_true)=3(三个尺度特征), y_true[i] = (3, 20, 20, cat_num+6), (3, 40, 40, cat_num+6), (3, 80, 80, cat_num+6)
        y_true = YOLOv5BestRatioAssigner(boxes, anchors=np.array(self.anchors), input_shape=self.input_shape, anchors_mask=self.anchors_mask, bbox_attrs=6+self.num_classes)
        return image.transpose(2,0,1), boxes, y_true
    



    def getDataByIndex(self, index):
        '''通过index获得图像, 图像的框, 以及框的标签
        Args:
            - index:  数据集里数据的索引
        Returns:
            - image:   训练集/测试集
            - box:   训练集/测试集
            - label:        categories字段映射
        '''         
        ann_file_name = self.ann_list[index]
        ann_path = os.path.join(self.ann_dir, ann_file_name)
        img_path = os.path.join(self.img_dir, ann_file_name.split('.')[0]+'.png')
        # 载入图像 (通过imgInfo获取图像名，得到图像路径)               
        image = Image.open(img_path)
        W, H = image.size
        image = np.array(image.convert('RGB'))
        boxes, labels, angle = [], [], []
        with open(ann_path, 'r') as ann_file:
            for line in ann_file.readlines():
                info = line[:-1].split(' ')
                x0, y0, x1, y1 = float(info[0]), float(info[1]), float(info[2]), float(info[3])
                x2, y2, x3, y3 = float(info[4]), float(info[5]), float(info[6]), float(info[7])
                poly = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], dtype=np.float32)
                # θ范围为(0, 90](有时候会取到0度，此时是错误的), 旧版本是[-90, 0)
                [(c_x, c_y), (w, h), theta] = cv2.minAreaRect(poly)  
                # 将rect转为长边表示法, [c_x, c_y, w, h, θ] -> [c_x, c_y, longside, shortside, θ] θ∈[-180, 0)
                # theta取负号是因为当θ范围为(0, 90]时,转换为旧版本的[-90, 0)
                long_side_format_rect = np.array(openCVFormat2LongsideFormat(c_x, c_y, w, h, -theta))
                cls = self.cat_names2id[info[8]]
                boxes.append(long_side_format_rect[:4])
                angle.append(long_side_format_rect[4])
                labels.append(cls)

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels)
        angle = np.array(angle, dtype=np.float32)

        return image, boxes, angle, labels



    def getDataByIndexYOLOLongSide(self, index):
        '''通过index获得图像, 图像的框, 以及框的标签(YOLOLongSide格式)
        Args:
            - index:  数据集里数据的索引
        Returns:
            - image:   训练集/测试集
            - box:   训练集/测试集
            - label:        categories字段映射
        '''         
        ann_file_name = self.ann_list[index]
        ann_path = os.path.join(self.ann_dir, ann_file_name)
        img_path = os.path.join(self.img_dir, ann_file_name.split('.')[0]+'.png')
        # 载入图像 (通过imgInfo获取图像名，得到图像路径)               
        image = Image.open(img_path)
        W, H = image.size
        image = np.array(image.convert('RGB'))
        boxes, labels, angle = [], [], []
        with open(ann_path, 'r') as ann_file:
            for line in ann_file.readlines():
                info = line[:-1].split(' ')
                cls, cx, cy, w, h, theta = int(info[0]), round(float(info[1])*W), round(float(info[2])*H), round(float(info[3])*W), round(float(info[4])*H), float(info[5])
                boxes.append([cx, cy, w, h])
                angle.append(theta)
                labels.append(cls)

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels)
        angle = np.array(angle, dtype=np.float32)

        return image, boxes, angle, labels
    




    def augment(self, image, boxes, angle, labels, mosaic_p=0.5):
        '''所有数据增强+预处理操作(顺序不能乱!)
        '''   
        if (self.mode):
            # 基本的数据增强
            image, boxes, angle = self.trainAlbumAug(image, boxes, angle)
            image, boxes, angle, labels, mosaic_flag = self.yoloMosaic4(image, boxes, angle, labels, p=mosaic_p)
            if mosaic_flag:
                # 如果经过mosaic增强，则旋转角度小一些(缓解padding里的box去不掉的现象)
                image, boxes, angle = self.tf.randomRotate(image, boxes, angle, 10, 1, p=0.5)
            else:
                # 如果未经过mosaic增强，则旋转角度大一些
                image, boxes, angle = self.tf.randomRotate(image, boxes, angle, 45, 1.3, p=0.5)
        # 数据预处理(归一化)
        image = self.normalAlbumAug(image)
        # 图像resize
        # image = self.tf.resize(image)
        return image, boxes, angle, labels


    def trainAlbumAug(self, image, boxes, angle):
        """基于albumentations库的训练时数据增强
        """
        # albumentation的图像维度得是[W,H,C]
        train_trans = self.tf.trainTF(image=image)
        image = train_trans['image'] 
        # 水平/竖直翻转增强
        image, boxes, angle = self.tf.hflip(image, boxes, angle, p=0.5)
        image, boxes, angle = self.tf.vflip(image, boxes, angle, p=0.5)
        image, boxes, angle = self.tf.mirrorflip(image, boxes, angle, p=0.5)
        # 这里的box是coco格式(xywh)
        return image, boxes, angle
        


    def normalAlbumAug(self, image):
        """基于albumentations库的基础数据预处理
        """
        normal_trans = self.tf.normalTF(image=image)
        image = normal_trans['image']
        # 这里的box是coco格式(xywh)
        return image
    



    def yoloMosaic4(self, image1, boxes1, angle1, labels1, scale=.5, p=0.5):
        """mosaic数据增强, 将四张图像拼在一起
        """
        mosaic_flag = False
        if (np.random.rand() < p):
            mosaic_flag = True
            # 随机选取其他3张图像的索引
            indexs = np.random.randint(self.datasetNum, size=3)
            # 读取其余3张图像, 对图像进行数据增强
            getData = {'dota':self.getDataByIndex, 'yolo':self.getDataByIndexYOLOLongSide}[self.ann_mode]
            image2, boxes2, angle2, labels2 = getData(indexs[0])
            image3, boxes3, angle3, labels3 = getData(indexs[1])
            image4, boxes4, angle4, labels4 = getData(indexs[2])
            image2, boxes2, angle2 = self.trainAlbumAug(image2, boxes2, angle2)
            image3, boxes3, angle3 = self.trainAlbumAug(image3, boxes3, angle3)
            image4, boxes4, angle4 = self.trainAlbumAug(image4, boxes4, angle4)
            images = [image1, image2, image3, image4]
            bboxes = [boxes1, boxes2, boxes3, boxes4]
            labels = [labels1, labels2, labels3, labels4]
            angles = [angle1, angle2, angle3, angle4]
            # mosaic4图像增强
            mosaic_img, bboxes, angles, labels = self.tf.yoloMosaic4(images, bboxes, angles, labels, scale=.5)
            return mosaic_img, bboxes, angles, labels, mosaic_flag
            
        return image1, boxes1, angle1, labels1, mosaic_flag






    # DataLoader中collate_fn参数使用
    # 由于检测数据集每张图像上的目标数量不一
    # 因此需要自定义的如何组织一个batch里输出的内容
    def dataset_collate(self, batch):
        images  = []
        bboxes  = []
        y_trues = [[] for _ in batch[0][2]]
        for img, box, y_true in batch:
            images.append(img)
            bboxes.append(box)
            for i, sub_y_true in enumerate(y_true):
                y_trues[i].append(sub_y_true)

        '''trick, 采样类别数量少的图片, 每个batch采样一次'''
        if self.sample_by_freq:
            # 根据训练集每个类别目标出现频率进行随机图片采样(sample_by_freq 目前还是只支持yolo格式处理)
            sampled_img, sampled_box, sampled_angle, sampled_label = sample_img_by_objfreq(self.img_dir, self.ann_dir, self.num_classes, self.cat_img_dict, self.sampling_ratio)
            sampled_img = cv2.cvtColor(sampled_img, cv2.COLOR_BGR2RGB)
            # 数据预处理与增强
            sampled_img, sampled_box, sampled_angle, sampled_label = self.augment(sampled_img, sampled_box.astype(np.float64), sampled_angle, sampled_label, mosaic_p=0)
            # 数据处理
            # cxcywh -> norm(cxcywh)
            sampled_box[:, [0, 2]] /= self.img_shape[1]
            sampled_box[:, [1, 3]] /= self.img_shape[0]
            sampled_boxes = np.concatenate((sampled_box, sampled_angle.reshape(-1, 1), sampled_label.reshape(-1, 1)), axis=1)
            # len(y_true)=3(三个尺度特征), y_true[i] = (3, 20, 20, cat_num+6), (3, 40, 40, cat_num+6), (3, 80, 80, cat_num+6)
            sampled_y_true = YOLOv5BestRatioAssigner(sampled_boxes, anchors=np.array(self.anchors), input_shape=self.input_shape, anchors_mask=self.anchors_mask, bbox_attrs=6+self.num_classes)
            images.append(sampled_img.transpose(2,0,1))
            bboxes.append(sampled_boxes)
            for i, sub_y_true in enumerate(sampled_y_true):
                y_trues[i].append(sub_y_true)
        # np -> tensor  
        images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        bboxes  = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
        y_trues = [torch.from_numpy(np.array(ann, np.float32)).type(torch.FloatTensor) for ann in y_trues]
        return images, bboxes, y_trues



    # 设置Dataloader的种子
    # DataLoader中worker_init_fn参数使
    # 为每个 worker 设置了一个基于初始种子和 worker ID 的独特的随机种子, 这样每个 worker 将产生不同的随机数序列，从而有助于数据加载过程的随机性和多样性
    @staticmethod
    def worker_init_fn(worker_id, seed, rank=0):
        worker_seed = rank + seed
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)











# 固定全局随机数种子
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def visBatch(batch, cat_names, theta_mode, showText=False):
    '''可视化训练集一个batch的数据
    Args:
        dataLoader: torch的data.DataLoader
    Retuens:
        None     
    '''
    cat_color = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0), (147, 116, 116), (0, 0, 255)]
    modify_theta = {'-180':-180.0, '-90':-90.0}[theta_mode]

    images, boxes = batch[0], batch[1]
    # 图像均值
    mean = np.array([0.485, 0.456, 0.406]) 
    # 标准差
    std = np.array([[0.229, 0.224, 0.225]]) 
    plt.figure(figsize = (10, 10))
    for idx, imgBoxLabel in enumerate(zip(images, boxes)):
        img, box = imgBoxLabel
        box = box.numpy()
        # norm(cxcywh) -> cxcywh
        box[:, [0, 2]] = box[:, [0, 2]] * img.shape[2]
        box[:, [1, 3]] = box[:, [1, 3]] * img.shape[1]
        ax = plt.subplot(8, 8, idx+1)
        img = img.numpy().transpose((1,2,0))
        # 由于在数据预处理时我们对数据进行了标准归一化，可视化的时候需要将其还原
        img = np.clip(img * std + mean, 0, 1)
        for instBox in box:
            rect = longsideFormat2OpenCVFormat(instBox[0], instBox[1], instBox[2], instBox[3], -instBox[4]+modify_theta)
            poly = np.float32(cv2.boxPoints(rect))

            # 显示框
            color = tuple([c/255 for c in cat_color[int(instBox[5])]])
            ax.add_patch(plt.Polygon(poly, edgecolor=color, facecolor='none', linewidth=0.6))
            # 显示类别
            if showText:
                ax.text(int(instBox[0]), int(instBox[1]), cat_names[int(instBox[5])], fontsize=3, bbox={'facecolor':'white', 'alpha':0.5, 'pad':1})
        plt.imshow(img)
        # 在图像上方展示对应的标签
        # 取消坐标轴
        plt.axis("off")
            # 微调行间距
        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    # plt.show()
    plt.savefig('plot_dataset.jpg', dpi=400)











def test_dota():
    # 固定随机种子 122
    seed = 11
    seed_everything(seed)
    sample_by_freq=True
    # BatcchSize
    BS = 64
    BS = BS - 1 if sample_by_freq else BS
    # 图像尺寸
    imgSize = [1024, 1024]
    input_shape = [1024, 1024]
    anchors = [[10, 13], [16, 30], [33, 23],
                [30, 61], [62, 45], [59, 119],
                [116, 90], [156, 198], [373, 326],
                ]
    anchors_mask = [[0,1,2], [3,4,5], [6,7,8]]
    ann_mode = 'yolo'
    theta_mode = '-180'


    '''DOTA-v1.0'''
    train_img_dir = "F:/Desktop/master/datasets/RemoteSensing/DOTA-1.0_ss_size-1024_gap-200/train/images"
    # train_ann_dir = "F:/Desktop/master/datasets/RemoteSensing/DOTA-1.0_ss_size-1024_gap-200/train/annfiles"
    train_ann_dir = 'F:/Desktop/master/datasets/RemoteSensing/DOTA-1.0_ss_size-1024_gap-200/train/yolo_longside_format_annfiles'
    cls_num = 15
    cat_names2id = {
        'plane':0, 'baseball-diamond':1, 'bridge':2, 'ground-track-field':3,
        'small-vehicle':4, 'large-vehicle':5, 'ship':6, 'tennis-court':7,
        'basketball-court':8, 'storage-tank':9, 'soccer-ball-field':10, 
        'roundabout':11, 'harbor':12, 'swimming-pool':13, 'helicopter':14
    }
    cat_names = ['PL', 'BD', 'BR', 'GTF', 'SV', 'LV', 'SH', 'TC', 'BC', 'ST', 'SBF', 'RA', 'HB', 'SP', 'HC']

    ''' 自定义数据集读取类'''
    trainDataset = DOTA2LongSideFormatYOLODataset(cls_num, cat_names2id, anchors, anchors_mask, train_ann_dir, train_img_dir, imgSize, input_shape, ann_mode, theta_mode=theta_mode, sample_by_freq=sample_by_freq)
    trainDataLoader = DataLoader(trainDataset, shuffle=True, batch_size=BS, num_workers=0, pin_memory=True,
                                    collate_fn=trainDataset.dataset_collate, worker_init_fn=partial(trainDataset.worker_init_fn, seed=seed))
    # validDataset = DOTA2LongSideFormatDataset(valAnnPath, valImgDir, imgSize, trainMode=False, map=map)
    # validDataLoader = DataLoader(validDataset, shuffle=True, batch_size=BS, num_workers=2, pin_memory=True, 
    #                               collate_fn=frcnn_dataset_collate, worker_init_fn=partial(worker_init_fn, seed=seed))



    print(f'训练集大小 : {trainDataset.__len__()}')
    cnt = 0
    for step, batch in enumerate(trainDataLoader):
        visBatch(batch, cat_names, theta_mode=theta_mode, showText=False)
        # box[i]: [:, 6], (cx, cy, long_side, short_side, θ, cls_id)
        images, boxes, y_trues = batch[0], batch[1], batch[2]
        cnt+=1
        # torch.Size([bs, 3, 800, 800])
        print(f'images.shape : {images.shape}')   
        # 列表形式，因为每个框里的实例数量不一，所以每个列表里的box数量不一
        print(f'len(boxes) : {len(boxes)}')     
        # 列表形式，因为每个框里的实例数量不一，所以每个列表里的label数量不一  
        print(f'len(labels) : {len(y_trues)}')  
        break














# for test only:
if __name__ == "__main__":
    test_dota()