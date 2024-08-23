import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.util import *
from utils.YOLOAnchorUtils import *
from loss.YOLOLoss import *






class YOLOv5Head(nn.Module):
    def __init__(self, l, cat_nums, img_size, anchors, in_channels, anchors_mask, reg_loss_type, cls_loss_type, box_loss_type, obj_loss_type, label_smoothing=0):
        '''RPN网络
        Args:

        Returns:
            None
        '''
        super(YOLOv5Head, self).__init__()
        # 当前head提取fpn哪一层特征(0:P3 1:P4 2:P5)
        self.l = l
        self.label_smoothing = label_smoothing
        self.anchors_mask = anchors_mask
        self.anchors = anchors
        self.img_size = img_size
        self.num_classes = cat_nums
        # 不同特征层的损失权重不同 p3,p4,p5, 对应大目标权重更低，小目标权重更高(只对obj_loss加权)
        self.balance = [4, 1.0, 0.4]
        # 自适应调整不同损失的权重，在COCO数据集下，默认回归损失权重0.05, 分类损失权重1 obj损失权重0.5
        self.box_ratio = 0.05
        self.obj_ratio = 1   #* (img_size[0] * img_size[1]) / (640 ** 2)
        self.cls_ratio = 0.5 #* (self.num_classes / 80)
        self.theta_ratio = 1

        '''损失函数'''
        self.reg_loss_type = reg_loss_type
        self.cls_loss_type = cls_loss_type
        self.box_loss_type = box_loss_type
        self.obj_loss_type = obj_loss_type
        self.clsLoss = Loss(loss_type=cls_loss_type)
        self.boxLoss = Loss(loss_type=box_loss_type)    
        self.objLoss = Loss(loss_type=obj_loss_type)    
        self.IoUSmoothl1Loss = IOUSmoothL1Loss()
        self.RIoULoss = RotatedIoULoss()

        '''coupled yolov5 head'''
        self.head = nn.Conv2d(in_channels, len(anchors_mask) * (6 + self.num_classes), 1)
        '''初始化权重'''
        init_weights(self.head, 'normal', 0, 0.01)




    def forward(self, x):
        '''前向传播
        '''
        predict = self.head(x) 
        return predict 



    def batchLoss(self, fpn_single_feat, y_true):
        input = self.forward(fpn_single_feat)
        #  获得bs，特征层的高和宽
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        # stride_h = stride_w = 32、16、8 (下采样率)
        stride_h = self.img_size[0] / in_h
        stride_w = self.img_size[1] / in_w
        # 此时获得的scaled_anchors大小是相对于特征层的
        scaled_anchors  = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        # torch.Size([bs, 255, w, h]) -> torch.Size([bs, 3, w, h, 85]) (85 : cx, cy, w, h, obj_score, cls_score=80)
        prediction = input.view(bs, len(self.anchors_mask[self.l]), self.num_classes+6, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        # cx, cy, w, h, obj_score, cls_score通过simoid限制到(0,1)之间
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = torch.sigmoid(prediction[..., 2]) 
        h = torch.sigmoid(prediction[..., 3]) 
        theta = torch.sigmoid(prediction[..., 4]) 
        conf = torch.sigmoid(prediction[..., 5])
        pred_cls = torch.sigmoid(prediction[..., 6:])

        # 将预测结果进行解码, 即将预测offset作用到anchors上(box和角度解耦开来分别计算损失, box损失依然按照水平框的逻辑)
        # pred_boxes.shape = torch.Size([bs, 3, w, h, 5]), 5=(cx, cy, w, h, θ)是相对于特征图尺寸的绝对坐标, 不是归一化坐标
        pred_boxes = YOLOv5Reg2Box(bs, self.l, x, y, h, w, theta, self.anchors_mask, scaled_anchors, in_h, in_w)
        y_true = y_true.type_as(x)
        total_loss, box_loss, cls_loss, theta_loss = 0, 0, 0, 0
        # 如果有正样本才继续
        if torch.sum(y_true[..., 5] == 1) != 0:
            # 正样本索引
            pos_idx = y_true[..., 5] == 1
            loss_theta = torch.tensor(0).to(x.device)
            if self.reg_loss_type == 'IoUSmoothl1Loss':
                '''角度回归损失(只对属于正样本的grid计算梯度)'''
                loss_theta = self.IoUSmoothl1Loss(pred_boxes, y_true[..., :5], pos_idx)
                '''定位损失(直接用的giou) [bs, 3, w, h](只对属于正样本的grid计算梯度)'''
                iou = bboxIoU(pred_boxes[..., :-1], y_true[..., :4], GIoU=True).type_as(x).squeeze(-1)
                loss_box = self.boxLoss(iou, y_true[..., 5])
            if self.reg_loss_type == 'RotatedIoU1Loss':
                '''角度+定位损失(rotated_iou)(只对属于正样本的grid计算梯度)'''
                loss_box, iou = self.RIoULoss(pred_boxes, y_true[..., :5], pos_idx)
            '''分类损失(只对属于正样本的grid计算梯度)'''
            # cls_gt.shape = [nums_gt, 10] (one-hot)
            cls_gt = smooth_labels(y_true[..., 6:][pos_idx], self.label_smoothing, self.num_classes)
            if self.cls_loss_type == 'BCELoss':
                loss_cls = self.clsLoss(pred_cls[pos_idx], cls_gt)
            if self.cls_loss_type == 'FocalLoss':
                loss_cls = torch.sqrt(self.clsLoss(prediction[..., 6:][pos_idx], cls_gt))
            theta_loss += loss_theta
            box_loss += loss_box
            cls_loss += loss_cls
            total_loss += loss_box * self.box_ratio + loss_cls * self.cls_ratio + loss_theta * self.theta_ratio 
            # obj正样本对应位置的预测值设置为这个位置的预测框与GT的giou(因此可以认为obj的GT其实是动态的?)
            tobj = torch.where(pos_idx, iou.detach().clamp(0), torch.zeros_like(y_true[..., 5]))
        else:
            tobj = torch.zeros_like(y_true[..., 5])
        '''目标损失(当前网格是否有目标)(对所有样本计算梯度)'''
        if self.obj_loss_type == 'BCELoss':
            obj_loss = self.clsLoss(conf, tobj)
        if self.obj_loss_type == 'FocalLoss':
            obj_loss = torch.sqrt(self.clsLoss(prediction[..., 5], tobj))
        
        obj_loss = obj_loss * self.balance[self.l] * self.obj_ratio
        total_loss += obj_loss
        '''loss统一为字典格式输出'''
        loss = dict(
            total_loss = total_loss,
            theta_loss = theta_loss,
            box_loss = box_loss,
            cls_loss = cls_loss,
            obj_loss = obj_loss
        )
        return loss
















# for test only
if __name__ == '__main__':
    from torchsummary import summary
    # 基本配置
    phi = 's'
    depth_dict          = {'n': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
    width_dict          = {'n': 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]
    base_channels       = int(wid_mul * 64)  # 64
    base_depth          = max(round(dep_mul * 3), 1)  # 3
    head_in_channel = {
        'n':[64,  128, 256 ],
        's':[128, 256, 512 ],
        'm':[192, 384, 768 ],
        'l':[256, 512, 1024],
        'x':[324, 640, 1280]
    }[phi]
    num_classes = 80 
    anchors_mask = [[0,1,2], [3,4,5], [6,7,8]]
    # head
    p3_head = YOLOv5Head(num_classes, base_channels * 4 , anchors_mask[0])
    p4_head = YOLOv5Head(num_classes, base_channels * 8 , anchors_mask[1])
    p5_head = YOLOv5Head(num_classes, base_channels * 16, anchors_mask[2])
    # 验证
    p3 = torch.rand((4, head_in_channel[0], 80, 80))
    p4 = torch.rand((4, head_in_channel[1], 40, 40))
    p5 = torch.rand((4, head_in_channel[2], 20, 20))
    p3_predict = p3_head(p3)
    p4_predict = p4_head(p4)
    p5_predict = p5_head(p5)
    # 
    print(p3_predict.shape) # torch.Size([4, 255, 80, 80])
    print(p4_predict.shape) # torch.Size([4, 255, 40, 40])
    print(p5_predict.shape) # torch.Size([4, 255, 20, 20])