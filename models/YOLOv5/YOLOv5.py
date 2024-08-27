import torch.nn as nn
from PIL import Image
from collections import Counter
import time

from utils.metrics import *
from models.YOLOv5.Backbone import *
from models.YOLOv5.PAFPN import *
from models.YOLOv5.Head import *
from utils.YOLOAnchorUtils import *


class Model(nn.Module):
    '''完整YOLOv5网络架构
    '''

    def __init__(self, backbone_name, img_size, anchors, anchors_mask, num_classes, phi, loadckpt, tta_img_size, backbone:dict, head:dict):
        super(Model, self).__init__()
        '''基本配置'''
        depth_dict          = {'n': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'n': 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]
        base_channels       = int(wid_mul * 64)  
        base_depth          = max(round(dep_mul * 3), 1)  
        # 类别数
        self.num_classes = num_classes
        self.img_size = img_size
        self.anchors = np.array(anchors)
        self.anchors_mask = anchors_mask
        '''网络基本组件'''
        self.backbone   = CSPDarknet(phi, base_channels, base_depth, **backbone)
        # self.backbone = Backbone(**backbone)
        self.fpn = YOLOv5PAFPN(base_channels, base_depth)
        self.p3_head = YOLOv5Head(0, self.num_classes, img_size, anchors, base_channels * 4 , anchors_mask, **head)
        self.p4_head = YOLOv5Head(1, self.num_classes, img_size, anchors, base_channels * 8 , anchors_mask, **head)
        self.p5_head = YOLOv5Head(2, self.num_classes, img_size, anchors, base_channels * 16, anchors_mask, **head)
        '''TTA增强'''
        self.tta = TTA(tta_img_size=tta_img_size)
        # 是否导入预训练权重
        if loadckpt: 
            print('ckpt path: ', loadckpt)
            self.load_state_dict(torch.load(loadckpt))
            # 基于尺寸的匹配方式(能克服局部模块改名加载不了的问题)
            # self = loadWeightsBySizeMatching(self, loadckpt)


    def forward(self, x):
        backbone_feat = self.backbone(x)
        p3, p4, p5 = self.fpn(backbone_feat)
        p3_predict = self.p3_head(p3)
        p4_predict = self.p4_head(p4)
        p5_predict = self.p5_head(p5)
        return p3_predict, p4_predict, p5_predict






    def batchLoss(self, device, img_size, batch_datas):
        '''一个batch的前向流程(不包括反向传播, 更新梯度)(核心, 要更改训练pipeline主要改这里)

        # Args:
            - `img_size`:     固定图像大小 如[832, 832]
            - `batch_imgs`:   一个batch里的图像              例:shape=[bs, 3, 600, 600]
            - `batch_bboxes`: 一个batch里的GT框              例:[(1, 4), (4, 4), (4, 4), (1, 4), (5, 4), (2, 4), (3, 4), (1, 4)]
            - `batch_labels`: 一个batch里的GT框类别          例:[(1,), (4,), (4,), (1,), (5,), (2,), (3,), (1,)]

        # # Returns:
            - losses: 所有损失组成的列表(里面必须有一个total_loss字段, 用于反向传播)
        '''
        batch_imgs, batch_bboxes, y_trues = batch_datas[0], batch_datas[1], batch_datas[2]
        batch_imgs = batch_imgs.to(device)
        batch_bboxes = [bboxes.to(device) for bboxes in batch_bboxes]
        # y_trues = [[...], [...], [...]]    y_trues[i].shape = [bs, 3, w, h, 5+cls_nums]
        y_trues = [y_true.to(device) for y_true in y_trues]
        # 前向过程
        backbone_feat = self.backbone(batch_imgs)
        p3, p4, p5 = self.fpn(backbone_feat)
        #  计算损失
        p3_loss = self.p3_head.batchLoss(p3, y_trues[0])
        p4_loss = self.p4_head.batchLoss(p4, y_trues[1])
        p5_loss = self.p5_head.batchLoss(p5, y_trues[2])
        loss = dict(
            total_loss = p3_loss['total_loss'] + p4_loss['total_loss'] + p5_loss['total_loss'],
            theta_loss = p3_loss['theta_loss'] + p4_loss['theta_loss'] + p5_loss['theta_loss'],
            box_loss = p3_loss['box_loss'] + p4_loss['box_loss'] + p5_loss['box_loss'],
            cls_loss = p3_loss['cls_loss'] + p4_loss['cls_loss'] + p5_loss['cls_loss'],
            obj_loss = p3_loss['obj_loss'] + p4_loss['obj_loss'] + p5_loss['obj_loss'],
        )
        return loss








    def infer(self, image:np.array, img_size, tf, device, T, image2color=None, agnostic=False, vis_heatmap=False, save_vis_path=None, half=False, tta=False):
        '''推理一张图/一帧
            # Args:
                - image:  读取的图像(nparray格式)
                - tf:     数据预处理(基于albumentation库)
                - device: cpu/cuda
                - T:      可视化的IoU阈值
            # Returns:
                - boxes:       网络回归的box坐标    [obj_nums, 4]
                - box_scores:  网络预测的box置信度  [obj_nums]
                - box_classes: 网络预测的box类别    [obj_nums]
        '''
        H, W = np.array(np.shape(image)[0:2])
        # tensor_img有padding的黑边
        # 注意permute(2,0,1) 不要写成permute(2,1,0)
        tensor_img = torch.tensor(tf.testTF(image=image)['image']).permute(2,0,1).unsqueeze(0).to(device)
        if half: tensor_img = tensor_img.half()
        with torch.no_grad():
            '''网络推理得到最原始的未解码未nms的结果'''
            # p3, p4, p5
            predicts = self.forward(tensor_img)
            '''利用Head的预测结果对RPN proposals进行微调+解码 获得预测框'''
            # torch.Size([1, 1200, 85])
            # torch.Size([1, 4800, 85])
            # torch.Size([1, 19200, 85])
            decode_predicts = inferDecodeBox(predicts, img_size, self.num_classes, self.anchors, self.anchors_mask)
            '''计算nms, 并将box坐标从归一化坐标转换为绝对坐标'''
            # torch.cat(decode_predicts, 1) : torch.Size([1, 25200, 85]) 
            # decode_predicts[0]: [num_anchors, 8] (8=cx, cy, w, h, theta, obj_score, cls_score, cls_id)
            decode_predicts = rotatedNMS(torch.cat(decode_predicts, 1), img_size, conf_thres=T, nms_thres=0.1, agnostic=agnostic)
            # 图像里没预测出目标的情况:
            if len(decode_predicts) == 0 : return [],[],[]
            box_classes = np.array(decode_predicts[0][:, 7], dtype = 'int32')
            box_scores = decode_predicts[0][:, 5] * decode_predicts[0][:, 6]
            # xywhθ
            boxes = decode_predicts[0][:, :5]
            '''box坐标映射(有灰边图像里的坐标->原图的坐标)'''
            # W, H 原始图像的大小
            H, W = image.shape[:2]
            max_len = max(W, H)
            # w, h 缩放后的图像的大小
            w = int(W * img_size[0] / max_len)
            h = int(H * img_size[1] / max_len)
            # 将box坐标(对应有黑边的图)映射回无黑边的原始图像
            boxes = mapBox2OriginalImg(boxes, W, H, [w, h], padding=True)
            '''是否可视化obj heatmap'''
            if vis_heatmap:vis_YOLOv5_heatmap(predicts, [W, H], img_size, image, box_classes, save_vis_path=save_vis_path)

            return boxes, box_scores, box_classes







    def onnxInfer(self, onnx_model, image:np.array, img_size, tf, device, T, image2color=None, agnostic=False, vis_heatmap=False, save_vis_path=None):
        '''推理一张图/一帧
            # Args:
                - image:  读取的图像(nparray格式)
                - tf:     数据预处理(基于albumentation库)
                - device: cpu/cuda
                - T:      可视化的IoU阈值
            # Returns:
                - boxes:       网络回归的box坐标    [obj_nums, 4]
                - box_scores:  网络预测的box置信度  [obj_nums]
                - box_classes: 网络预测的box类别    [obj_nums]
        '''
        H, W = np.array(np.shape(image)[0:2])
        # tensor_img有padding的黑边
        # 注意permute(2,0,1) 不要写成permute(2,1,0)
        tensor_img = torch.tensor(tf.testTF(image=image)['image']).permute(2,0,1).unsqueeze(0)
        onnx_input_img = tensor_img.numpy()

        '''网络推理得到最原始的未解码未nms的结果'''
        # p3, p4, p5
        p3_predict, p4_predict, p5_predict = onnx_model.run(['p3_head', 'p4_head', 'p5_head'], {'input': onnx_input_img})
        predicts = [torch.tensor(p3_predict).to(device), torch.tensor(p4_predict).to(device), torch.tensor(p5_predict).to(device)]
        '''利用Head的预测结果对RPN proposals进行微调+解码 获得预测框'''
        # torch.Size([1, 1200, 85])
        # torch.Size([1, 4800, 85])
        # torch.Size([1, 19200, 85])
        decode_predicts = inferDecodeBox(predicts, img_size, self.num_classes, self.anchors, self.anchors_mask)
        '''计算nms, 并将box坐标从归一化坐标转换为绝对坐标'''
        # torch.cat(decode_predicts, 1) : torch.Size([1, 25200, 85])
        decode_predicts = rotatedNMS(torch.cat(decode_predicts, 1), img_size, conf_thres=T, nms_thres=0.1, agnostic=agnostic)
        # 图像里没预测出目标的情况:
        if len(decode_predicts) == 0 : return [],[],[]
        box_classes = np.array(decode_predicts[0][:, 7], dtype = 'int32')
        box_scores = decode_predicts[0][:, 5] * decode_predicts[0][:, 6]
        # xyxy
        boxes = decode_predicts[0][:, :5]
        '''box坐标映射(有灰边图像里的坐标->原图的坐标)'''
        # W, H 原始图像的大小
        H, W = image.shape[:2]
        max_len = max(W, H)
        # w, h 缩放后的图像的大小
        w = int(W * img_size[0] / max_len)
        h = int(H * img_size[1] / max_len)
        # 将box坐标(对应有黑边的图)映射回无黑边的原始图像
        boxes = mapBox2OriginalImg(boxes, W, H, [w, h], padding=True)
        '''是否可视化obj heatmap'''
        if vis_heatmap:vis_YOLOv5_heatmap(predicts, [W, H], img_size, image, box_classes, save_vis_path=save_vis_path)

        return boxes, box_scores, box_classes
        










# for test only
if __name__ == '__main__':
    phi='x'
    model = Model(anchors_mask=[[6,7,8], [3,4,5], [0,1,2]], num_classes=80, phi=phi, pretrained=False, loadckpt=None)
    torch.save(model.state_dict(), f"{phi}.pt")
    # 验证 1
    # summary(model, input_size=[(3, 224, 224)])  
    # 验证 2
    x = torch.rand((8, 3, 640, 640))
    p3_predict, p4_predict, p5_predict = model(x)
    print(p3_predict.shape)    
    print(p4_predict.shape)   
    print(p5_predict.shape) 
