import albumentations as A
import numpy as np
import random
import cv2






class Transform():
    '''数据预处理/数据增强(基于albumentations库)
       https://albumentations.ai/docs/api_reference/full_reference/
    '''
    def __init__(self, box_format:str='coco', img_shape:list[int]=[1024,1024], input_shape:list[int]=[1024,1024], dataset_num:int=None):
        ''' 
            - img_shape:   图像大小
            - input_shape: 网络接受的输入大小
            - box_format:  'yolo':norm(cxcywh), 'coco':xywh, cxcywh, norm(cxcywh)
        '''
        self.img_shape = img_shape
        self.input_shape = input_shape
        self.dataset_num = dataset_num
        # 训练时增强
        self.trainTF = A.Compose([
                # 参数：随机色调、饱和度、值变化
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
                # 随机对比度增强
                A.CLAHE(p=0.1),
                # 高斯噪声
                A.GaussNoise(var_limit=(0.05, 0.09), p=0.4),     
                # 随机转为灰度图
                A.ToGray(p=0.01),
                # 随机DropBlock
                A.CoarseDropout(max_holes=60, max_height=25, max_width=25, min_holes=30, min_height=5, min_width=5, fill_value=128, p=0.0),
                A.OneOf([
                    # 使用随机大小的内核将运动模糊应用于输入图像
                    A.MotionBlur(p=0.2),   
                    # 中值滤波
                    A.MedianBlur(blur_limit=3, p=0.1),    
                    # 使用随机大小的内核模糊输入图像
                    A.Blur(blur_limit=3, p=0.1),  
                ], p=0.2),
                
            ],
            )
        # 基本数据预处理
        self.normalTF = A.Compose([
                # # 最长边限制为imgSize
                # A.LongestMaxSize(max_size=maxSize),
                # # 较短的边做padding
                # A.PadIfNeeded(imgSize[0], imgSize[1], border_mode=cv2.BORDER_CONSTANT, value=[128,128,128]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
            )
        # 基本数据预处理
        self.testTF = self.normalTF      


    def resize(self, image):
        """图像resize
        """
        if (self.img_shape != self.input_shape):
            image = cv2.resize(image, dsize=self.input_shape)
        return image


    def hflip(self, image, boxes, angle, p):
        """水平翻转增强
        """
        if (np.random.rand() < p):
            image = cv2.flip(image, 1)
            boxes[:, 0] = self.img_shape[0] - boxes[:, 0]
            angle = - angle - 180
        return image, boxes, angle


    def vflip(self, image, boxes, angle, p):
        """竖直翻转增强
        """
        if (np.random.rand() < p):
            image = cv2.flip(image, 0)
            boxes[:, 1] = self.img_shape[1] - boxes[:, 1]
            angle = - angle - 180
        return image, boxes, angle
    

    def mirrorflip(self, image, boxes, angle, p):
        """斜对称翻转增强
        """
        if (np.random.rand() < p):
            image = cv2.transpose(image)
            boxes[:, [0, 1]] = boxes[:, [1, 0]]
            angle = -270 - angle
            angle[angle < -180] += 180
        return image, boxes, angle
    


    def randomRotate(self, image, boxes, angle, aug_angle, ratio, p=0.5):
        """随机旋转增强
        """
        if (np.random.rand() < p):
            W, H = self.img_shape
            # 绕图片中心进行旋转
            center = (W / 2, H / 2)   
            # 旋转方向取(-aug_angle，aug_angle)中的随机整数值，负为逆时针，正为顺势针
            aug_angle = random.randint(-aug_angle, aug_angle)  
            scale = np.cos(aug_angle*np.pi/180)*ratio
            # 获得旋转矩阵
            M = cv2.getRotationMatrix2D(center, aug_angle, scale)
            # 对cxcy执行旋转变换
            cxcy = np.concatenate((boxes[:, :2].T, np.ones((1, boxes.shape[0]))))
            rotate_cxcy = (M @ cxcy).T
            boxes[:, :2] = rotate_cxcy
            # 对wh进行缩放
            boxes[:, 2:] *= scale
            angle += aug_angle
            # 对超出范围的角度进行调整
            angle[angle < -180] += 180
            angle[angle > 0] -= 180
            # 进行仿射变换，边界填充为128
            image = cv2.warpAffine(src=image, M=M, dsize=(H, W), borderValue=(128, 128, 128))

        return image, boxes, angle




    def yoloMosaic4(self, images, bboxes, angles, labels, scale=.5):
        """mosaic数据增强, 将四张图像拼在一起
        """

        W, H = self.input_shape
        # 放置图像的中心位置
        cx = int(random.uniform(0.3, 0.7) * W)
        cy = int(random.uniform(0.3, 0.7) * H)
        mosaic_img = np.ones((W, H, 3), dtype=np.uint8) * 128
        for i in range(4):
            bboxes[i] = np.array(bboxes[i])
            labels[i] = np.array(labels[i])
            w, h, _ = images[i].shape
            # 对图像进行缩放并且进行长和宽的扭曲
            scale = random.uniform(scale, 1)
            new_w, new_h = int(w * scale), int(h * scale)
            # 对图像进行缩放
            images[i] = cv2.resize(images[i], (new_h, new_w))
            # 对box进行缩放
            bboxes[i] *= scale
            # 图像mosaic到一张图像上:
            if i==0: 
                mosaic_img[max(cx-new_w, 0):cx, max(cy-new_h, 0):cy, :] = images[i][max(0, new_w-cx):, max(0, new_h-cy):, :]
                # 对图像进行平移
                bboxes[i][:,0] += (cy-new_h)
                bboxes[i][:,1] += (cx-new_w)
            if i==1:
                mosaic_img[cx:min(W, cx+new_w), max(cy-new_h, 0):cy, :] = images[i][:min(new_w, W-cx), max(0, new_h-cy):, :]
                # 对图像进行平移
                bboxes[i][:,0] += (cy-new_h)
                bboxes[i][:,1] += cx
            if i==2: 
                mosaic_img[max(cx-new_w, 0):cx, cy:min(H, cy+new_h), :] = images[i][max(0, new_w-cx):, :min(new_h, H-cy), :]
                # 对图像进行平移
                bboxes[i][:,0] += cy
                bboxes[i][:,1] += (cx-new_w)
            if i==3: 
                # 对图像进行平移
                bboxes[i][:,0] += cy
                bboxes[i][:,1] += cx
                mosaic_img[cx:min(W, cx+new_w), cy:min(H, cy+new_h), :] = images[i][:min(new_w, W-cx), :min(new_h, H-cy), :]

            # 边界处理
            # keep = np.where(np.logical_and(bboxes[i][:,0]>-20, bboxes[i][:,1]>-20))[0]
            # bboxes[i] = bboxes[i][keep]
            # labels[i] = labels[i][keep]
            # angles[i] = angles[i][keep]
            # keep = np.where(np.logical_and((bboxes[i][:,2] + bboxes[i][:,0])<self.img_shape[0]+20, (bboxes[i][:,3] + bboxes[i][:,1])<self.img_shape[1]+20))[0]
            # bboxes[i] = bboxes[i][keep]
            # labels[i] = labels[i][keep]
            # angles[i] = angles[i][keep]

        labels = np.concatenate(labels, axis=0)
        angles = np.concatenate(angles, axis=0)
        bboxes = np.concatenate(bboxes, axis=0)
        if len(bboxes) != 0:
            return mosaic_img, bboxes, angles, labels
 
        return images, bboxes, angles, labels