import torch
import torch.nn as nn
from mmcv.ops import box_iou_quadri, box_iou_rotated
# diff_iou_rotated_2d会保留梯度信息
from mmcv.ops import diff_iou_rotated_2d



def clip_by_tensor(t, t_min, t_max):
    # 对预测结果做数值截断
    t = t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def MSELoss(pred, target):
    return torch.pow(pred - target, 2)

def BCELoss(pred, target, epsilon=1e-7):
    pred    = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output  = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


def FocalLoss(pred, target, loss_fcn, gamma=1.5, alpha=0.25):
    loss = loss_fcn(pred, target)
    # p_t = torch.exp(-loss)
    # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

    # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
    pred_prob = torch.sigmoid(pred)  # prob from logits
    p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    modulating_factor = (1.0 - p_t) ** gamma
    loss *= alpha_factor * modulating_factor
    return loss


def QFocalLoss(pred, target, loss_fcn, gamma=1.5, alpha=0.25):
    loss = loss_fcn(pred, target)

    pred_prob = torch.sigmoid(pred)  # prob from logits
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    modulating_factor = torch.abs(target - pred_prob) ** gamma
    loss *= alpha_factor * modulating_factor
    return loss


def GIoULoss(box_giou, target):
    # 定位损失(直接用的giou)
    loss = (1 - box_giou)[target == 1]
    return torch.mean(loss)










class Loss(nn.Module):
    def __init__(self, loss_type:str, gamma=1.5, alpha=0.25, reduction='mean'):
        '''
            - loss_type: BCELoss | MSELoss | FocalLoss | QFocalLoss | GIoULoss
        '''
        super().__init__()
        self.reduction = reduction
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")
        self.loss_type = loss_type
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        if self.loss_type == 'BCELoss':
            loss = BCELoss(pred, target)
        elif self.loss_type == 'MSELoss':
            loss = MSELoss(pred, target)
        elif self.loss_type == 'FocalLoss':
            loss = FocalLoss(pred, target, self.loss_fcn, self.gamma, self.alpha)
        elif self.loss_type == 'QFocalLoss':
            loss = QFocalLoss(pred, target, self.loss_fcn, self.gamma, self.alpha)
        elif self.loss_type == 'GIoULoss':
            loss = GIoULoss(pred, target)
        
        if self.reduction=='mean':
            return torch.mean(loss)
        if self.reduction=='sum':
            return torch.sum(loss)
        if self.reduction=='none':
            return loss





class IOUSmoothL1Loss(nn.Module):
    '''基于rIoU加权的角度回归SmoothL1损失
    # 基本思想
    - SmoothL1提供角度回归的梯度方向, rIoU提供梯度的大小(当角度预测接近边界值时, rIoU很小,能够有效缓解损失函数在边界突变的情况)
    '''
    def __init__(self):
        super().__init__()
        self.basic_loss = nn.SmoothL1Loss(reduction="none")
        # 防止除以0的情况出现
        self.e = 1e-7
        self.theta_T = 10

    def computeFactor(self, basic_loss, pred_rboxes, target_rboxes):
        '''factor和rIoU相关
        '''
        with torch.no_grad():
            # 将GT的θ转为原始的θ(使用clone是为了避免原地操作):
            target_rboxes_copy = target_rboxes.clone()
            # 注意, mmcv.ops.box_iou_rotated接受的角度为弧度制, 需要将角度转化为弧度!!
            target_rboxes_copy[:, -1] = torch.deg2rad(target_rboxes_copy[:, -1] * 180 - 180)
            pred_rboxes_copy = pred_rboxes.clone()  # 克隆pred_rboxes以避免就地操作
            pred_rboxes_copy[:, -1] = torch.deg2rad(pred_rboxes_copy[:, -1] * 180 - 180)
            # 将riou作为损失的大小
            riou = box_iou_rotated(pred_rboxes_copy, target_rboxes_copy, aligned=True, clockwise=True)
            # 限制riou的范围不超过(1e-7, 1)
            riou = torch.clamp(riou, min=self.e, max=1 - 2 * self.e)
            # 将损失归一化, 只保留梯度的方向:
            norm_factor = basic_loss.abs() + self.e
            factor = 0.005 * riou.log().abs() / norm_factor
            '''对于那些GT不是在边界范围内的框, 则还是使用原本的SmoothL1的梯度, 不使用IoU'''
            target_theta = target_rboxes[:, -1] * 180 - 180
            # 角度超出边界范围10度以上就不使用rIoU因子
            condition = (target_theta > -180 + self.theta_T) & (target_theta < -self.theta_T)
            factor[condition] = 1.
            return factor

    def forward(self, pred_theta, target_theta, pred_hboxes, target_hboxes):
        '''pred_hboxes, target_hboxes的框坐标均是基于特征图尺寸下的绝对坐标, 
           pred_theta和target_theta均为归一化的θ
        '''
        basic_loss = self.basic_loss(pred_theta, target_theta)
        pred_rboxes = torch.cat([pred_hboxes, pred_theta.unsqueeze(1)], dim=1)
        target_rboxes = torch.cat([target_hboxes, target_theta.unsqueeze(1)], dim=1)
        '''计算加权因子'''
        factor = self.computeFactor(basic_loss, pred_rboxes, target_rboxes)
        # 在SmoothL1损失的基础之上乘一个加权因子(和IoU有关)
        loss = basic_loss * factor
        return loss.mean()
    

    



    




class RotatedIoULoss(nn.Module):
    """RotatedIoULoss.
        https://github.com/open-mmlab/mmrotate/blob/9ea1aeeef2da8b2cd5161b72f4e33e1e8293dcb2/mmrotate/models/losses/rotated_iou_loss.py
    """

    def __init__(self, eps=1e-7, mode='linear'):
        '''
        - mode: iou loss的形式
        '''
        super(RotatedIoULoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        self.mode = mode
        self.eps = eps

    def forward(self, pred, target, pos_idx):
        """Rotated IoU loss.

        Args:
            pred (torch.Tensor): Predicted bboxes of format (x, y, h, w, angle),
                shape (n, 5).
            target (torch.Tensor): Corresponding gt bboxes, shape (n, 5).
            pos_idx: 正样本的索引
        """
        
        # 注意, mmcv.ops.diff_iou_rotated_2d接受的角度为弧度制, 需要将角度转化为弧度!!
        bs, anchor_num, w, h = target.shape[:-1]
        target[..., -1] = torch.deg2rad(target[..., -1] * 180 - 180)
        pred[..., -1] = torch.deg2rad(pred[..., -1])
        # 注意角度是弧度
        ious = diff_iou_rotated_2d(pred.reshape(1, -1, 5), target.reshape(1, -1, 5))
        # reshape回原来的形状
        ious = ious.reshape(bs, anchor_num, w, h).clamp(min=self.eps, max=1.0-self.eps)
        pos_ious = ious[pos_idx]
        # NOTE:用linear
        if self.mode == 'linear':
            loss = 1 - pos_ious
        elif self.mode == 'square':
            loss = 1 - pos_ious**2
        elif self.mode == 'log':
            loss = -pos_ious.log()
        else:
            raise NotImplementedError
        # 返回loss和riou(计算obj_loss用得到)
        return loss.mean(), ious