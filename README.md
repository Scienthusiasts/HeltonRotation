# HeltonRotation
从0开始，搭建一个深度学习有向目标检测框架！(基于Pytorch，不定期更新) 

该项目的基本框架逻辑遵循：[HeltonDetection](https://github.com/Scienthusiasts/HeltonDetection)

**Create 24/7/21**

- 发布dev分支

- 将YOLOv5改为旋转框基本思路：

> 目前的主要修改逻辑是, 网络的box部分和角度部分的回归完全解耦开来, 独立进行回归和计算损失, anchor的形式也不用变
> 对于box回归部分,把box完全当成水平框去对待(直接就是旋转宽的wh,而不是其外接水平框的wh), box依然计算水平框的iou损失(和水平框完全一致不用更改), 不会用到skewIoU
> 且正负样本分配的基本逻辑也不用更改, 还是完全基于YOLOv5水平框的那一套逻辑
> 对于角度回归的部分, 目前先测试一下直接采用SmoothL1Loss进行回归能否收敛(会存在明显的边界问题), 之后再尝试其他的方法
> [如何缓解角度的边界问题:从损失函数入手; 从旋转目标的定义入手,定义一种新的不存在便捷问题的有向目标表示方式]
>
> **1.修改数据集读取的方式:**
> 本项目支持基于DOTA格式的8参表示法和基于YOLO格式的5参长边表示法, 目前默认角度范围为[-180, 0), 
> 对于DOTA格式，首先会使用opencv的最小外接旋转矩形的函数处理转化为五参表示法, 最后都会统一转为YOLO的长边格式
> 自定义实现了旋转,翻转,rotatedMosaic等增强方式(因为Albumentation库好像不支持旋转框的增强), rotatedMosaic增强还有点小Bug,可能会出现超出图像之外的框消除不掉的情况
> 此外, YOLOv5的正负样本分配策略是静态的, 这部分主要添加y_trues中角度参数的部分, 同时将角度GT归一化到0,1之间
>
> **2.修改网络部分:**
> 其实主要是Head部分,Backbone和Neck可以完全不动, Head主要针对添加角度回归这一参数进行修改
> 网络结构部分,添加对角度的回归, 回归的是归一化到0,1的角度(目前是对回归的结果取sigmoid, 先看下能不能收敛)
> 对于BatchLoss函数，主要添加角度回归损失
>
> **3.修改评估,测试部分的逻辑:**
> 测试和评估涉及到需要计算nms(训练的时候没用到NMS),因此对于这部分的修改,需要引入旋转IoU和旋转框的NMS,这部分我直接基于mmcv.ops中的rotated算子进行实现
> 对于测试一张图像, 只需要将预测的theta作用到回归框的坐标上即可,这里主要修改可视化函数，添加rbox2PolyNP方法将5参表示法转化为8参表示法,再利用cv2.drawContours进行绘制
> 对于评估整个验证集,这里遵循DOTA_devikit的评估方式, 即将模型的预测结果按照类别分别保存在相应类别的txt文档当中(格式为: file_name, score, 8个坐标), 评估时直接调用DOTA_devikit下dota_evaluation_task1.voc_eval方法

**如何在test上测试并提交DOTA服务器**

> 目前的方法，以后可以修改整合
>
> 将config文件下的`val_img_dir`字段设置为test图像所在目录，然后进行测试，报错了不用管,因为test没有GT文件
>
> 在`utils\metrics.py`下执行merge对应方法即得到整合后的模型推理结果(将split上的推理结果合并)
>
> 在https://captain-whu.github.io/DOTA/evaluation.html提交



**update 24/9/9**

- 支持FCOS_obb
- 添加了sampled_by_freq 数据采样策略
- 支持可视化置信度热力图(centerness×score)

**update 24/9/18**

- 添加mrecall, mprecision评估指标（tensorboard, 日志）

**update 24/9/20**

- 添加partial label设置(对比半监督)

- 针对FCOS-obb，添加了高斯椭圆正负样本分配策略

**update 24/10/1**

- 支持更多学习率衰减策略

**update 24/10/12**

- add prototypes for experimental validate

## Demo

![1](https://github.com/Scienthusiasts/HeltonRotation/blob/dev/demo/demo.png)

## Environments

```
整理与完善中...
```



## Train/Eval/Test

```
整理与完善中...
```



## Experiment

基于`DOTA_devkit`提供的接口进行评估(**mAP基于VOC07，NMSIoU阈值0.1(大于过滤)，置信度阈值0.01**)；默认使用warmup+cos学习率衰减策略

**`selective_IoU_smooth_l1`和`IoU_smooth_l1`有什么区别**：IoU_smooth_l1源自论文SCRDet里使用的损失函数，而selective将那些角度不在边界范围内(10度)的角度依然用smoothl1损失，边界范围内才使用IoU_smooth_l1损失

### YOLOv5_obb

- `DOTA-v1.0` (训练集为train-split，表格为val-split上的评估结果)

image-size=[1024, 1024]

batch-size=16

|        Model         |           theta_loss           | cls_loss   | optim | epoch | max_lr | lr_decay |  mAP50(%)  |
| :------------------: | :----------------------------: | ---------- | :---: | :---: | :----: | :------: | :--------: |
|       YOLOv5s        |           smooth_l1            | BCE loss   | adamw |  49   |  1e-3  |   0.1    |   58.434   |
|       YOLOv5s        |           smooth_l1            | BCE loss   | adamw |  101  |  1e-3  |   0.1    |   63.003   |
|       YOLOv5s        |         IoU_smooth_l1          | BCE loss   | adamw |  101  |  1e-3  |   0.1    |   62.435   |
|       YOLOv5s        |    selective_IoU_smooth_l1     | BCE loss   | adamw |  101  |  1e-3  |   0.1    |   65.427   |
|       YOLOv5s        |    selective_IoU_smooth_l1     | BCE loss   | adamw |  101  |  1e-3  |   0.01   |   64.177   |
|       YOLOv5s        |    selective_IoU_smooth_l1     | focal loss | adamw |  101  |  1e-3  |   0.1    |   65.430   |
|       YOLOv5s        |         RIoU_loss(box)         | focal loss | adamw |  101  |  1e-3  |   0.1    |   59.724   |
|       YOLOv5s        |      RIoU_loss(box, obj)       | focal loss | adamw |  101  |  1e-3  |   0.1    |   66.529   |
|       YOLOv5s        | RIoU_loss_**linear**(box, obj) | focal loss | adamw |  101  |  1e-3  |   0.1    |            |
|       YOLOv5s        |    selective_IoU_smooth_l1     | focal loss |  sgd  |  101  |  1e-2  |   0.1    |   65.271   |
|       YOLOv5s        |         RIoU_loss(box)         | focal loss |  sgd  |  101  |  1e-2  |   0.1    |   64.401   |
|       YOLOv5s        |      RIoU_loss(box, obj)       | focal loss |  sgd  |  101  |  1e-2  |   0.1    |   65.606   |
| YOLOv5s-COCOPretrain |    selective_IoU_smooth_l1     | focal loss | adamw |  101  |  1e-3  |   0.1    |   63.770   |
| YOLOv5l (train_ddp)  |    selective_IoU_smooth_l1     | focal loss | adamw |  101  |  1e-3  |   0.1    |   64.565   |
|       YOLOv5l        |         IoU_smooth_l1          | focal loss |  sgd  |  101  |  1e-2  |   0.1    |   70.192   |
| YOLOv5l (train_ddp)  |    selective_IoU_smooth_l1     | focal loss |  sgd  |  101  |  1e-2  |   0.1    | **71.255** |
| YOLOv5l (train_ddp)  |      RIoU_loss(box, obj)       | focal loss |  sgd  |  101  |  1e-2  |   0.1    |   69.234   |
| YOLOv5l (train_ddp)  |      RIoU_loss(box, obj)       | focal loss | adamw |  101  |  1e-3  |   0.1    |   66.389   |
| YOLOv5l (train_ddp)  |    selective_IoU_smooth_l1     | BCE loss   |  sgd  |  101  |  1e-2  |   0.1    |   69.822   |

**upload to DOTA server evaluation result (testset):**

yolov5s_IoUsmooths1_rootfocalloss_epoch101_lr1e-2_sgd_trainval

|  PL   |  BD   |  BR   |  GTF  |  SV   |  LV   |  SH   |  TC   |  BC   |  ST   |  SBF  |  RA   |  HA   |  SP   |  HC   | <font color=Red>mAP50</font> | mAP75 |  mAP  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--------------------------: | :---: | :---: |
| 87.62 | 75.27 | 31.22 | 58.92 | 76.47 | 73.97 | 80.51 | 90.79 | 64.59 | 86.06 | 29.86 | 59.03 | 56.56 | 68.37 | 42.38 |          **65.44**           | 30.07 | 33.63 |

yolov5s_Select_IoUsmooths1_rootfocalloss_epoch101_lr1e-2_sgd_trainval

|  PL   |  BD   |  BR   |  GTF  |  SV   |  LV   |  SH   |  TC   |  BC   |  ST   |  SBF  |  RA   |  HA   |  SP   |  HC   | <font color=Red>mAP50</font> | mAP75 |  mAP  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--------------------------: | :---: | :---: |
| 87.44 | 76.67 | 35.72 | 59.60 | 76.78 | 75.14 | 83.80 | 90.87 | 77.74 | 86.29 | 46.21 | 59.50 | 63.47 | 70.85 | 52.81 |          **69.52**           | 34.28 | 37.12 |

yolov5l_Select_IoUsmooths1_rootfocalloss_epoch101_lr1e-2_sgd_trainval

|  PL   |  BD   |  BR   |  GTF  |  SV   |  LV   |  SH   |  TC   |  BC   |  ST   |  SBF  |  RA   |  HA   |  SP   |  HC   | <font color=Red>mAP50</font> | mAP75 |  mAP  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--------------------------: | :---: | :---: |
| 88.26 | 81.62 | 43.97 | 60.45 | 78.99 | 77.09 | 86.71 | 90.82 | 87.08 | 87.24 | 52.79 | 65.46 | 67.96 | 72.22 | 54.52 |          **73.01**           | 40.09 | 41.84 |

### FCOS_obb

FCOS改为旋转框：在regression分支加了角度回归头(角度回归和坐标回归解耦)

- `DOTA-v1.0` (**训练集为train-split，表格为val-split上的评估结果(保留没有GT的图像)**)

- image-size=[1024, 1024]， **gap=200**

- **batch-size=8**

- epoch=37

- lr_decay=0.1

|                        Model                        | theta_loss_w |       theta_loss        |  bs  | optim | max_lr |       mAP50(%)       |
| :-------------------------------------------------: | :----------: | :---------------------: | :--: | :---: | :----: | :------------------: |
|                        FCOS                         |      1       | selective_IoU_smooth_l1 |  8   |  sgd  | 2.5e-3 |        59.015        |
|                        FCOS                         |      1       | selective_IoU_smooth_l1 |  8   | adamw |  2e-4  |        66.282        |
|                        FCOS                         |      1       | selective_IoU_smooth_l1 |  8   | adamw |  1e-3  |        68.928        |
|                        FCOS                         |      10      | selective_IoU_smooth_l1 |  8   | adamw |  2e-4  |        66.290        |
|                        FCOS                         |      1       |   Rotated_IoU(linear)   |  8   | adamw |  1e-3  |        69.360        |
|                        FCOS                         |      1       |   Rotated_IoU(linear)   |  4   | adamw |  1e-3  |        69.030        |
|                        FCOS*                        |      1       |   Rotated_IoU(linear)   |  8   | adamw |  1e-3  |        70.455        |
|                        FCOS*                        |      1       |   Rotated_IoU(linear)   |  4   | adamw |  1e-3  |        68.790        |
|                 FCOS-reg_centerness                 |      1       |   Rotated_IoU(linear)   |  8   | adamw |  1e-3  |    69.655/69.501     |
|                        FCOS                         |      1       |   Rotated_IoU(linear)   |  8   |  sgd  | 2.5e-3 |        61.513        |
|                        FCOS                         |      1       |   Rotated_IoU(linear)   |  8   |  sgd  |  1e-2  |        62.854        |
|                       FCOS-GA                       |      1       |   Rotated_IoU(linear)   |  8   | adamw |  1e-3  |        70.148        |
|               FCOS-reg_centerness-GA                |      1       |   Rotated_IoU(linear)   |  8   | adamw |  1e-3  | 70.648/70.835(A6000) |
|               FCOS-reg_centerness-GA*               |      1       |   Rotated_IoU(linear)   |  8   | adamw |  1e-3  |  **70.929**(A6000)   |
|  FCOS-reg_centerness-GA-w/o-mask-in-gt-box-center1  |      1       |   Rotated_IoU(linear)   |  8   | adamw |  1e-3  |        70.096        |
| FCOS-reg_centerness-GA-w/o-mask-in-gt-box-center0.7 |      1       |   Rotated_IoU(linear)   |  8   | adamw |  1e-3  |        70.540        |
| FCOS-reg_centerness-GA-w/o-mask-in-gt-box-center0.5 |      1       |   Rotated_IoU(linear)   |  8   | adamw |  1e-3  |        69.850        |
|           FCOS-reg_centerness-GA-more-pos           |      1       |   Rotated_IoU(linear)   |  8   | adamw |  1e-3  |        70.426        |
|       FCOS-reg_centerness-GA-cls_pos_cnt_all        |      1       |   Rotated_IoU(linear)   |  8   | adamw |  1e-3  |        69.338        |
|        FCOS-reg_centerness-GA-clshead1x1conv        |      1       |   Rotated_IoU(linear)   |  8   | adamw |  1e-3  |    69.891/69.670     |
|               FCOS-reg_centerness-10%               |      1       |   Rotated_IoU(linear)   |  8   | adamw |  1e-3  |        53.479        |
|               FCOS-reg_centerness-10%               |      1       |   Rotated_IoU(linear)   |  8   |  sgd  | 2.5e-3 |        49.910        |

*表示 sample img by categories frequency， 即根据数据集中每个类别下的目标GT数量的多少计算采样比例(**根据比例取倒数作为采样概率**)，GT数量越少的类别就有越大的概率采样到，采样到的图片包含对应类别的GT，同时，在采样的图片上，还会将那些GT数量较多的类别的GT进行mask(只保留GT数量最少的3个类别)，其中**每个batch里有一张是采样的图像**。

GA表示高斯椭圆分配策略，cls和centerness分支采用高斯椭圆的分配方法(回归分支仍进一步通过半径限制正样本范围)，相比原始FCOS的分配策略引入了目标角度的信息，更适合有向目标检测任务。

GA-w/o-mask-in-gt-box表示不加mask-in-gt-box这个约束(这个本是用来约束gt是否在矩形框内部的，其实有了mask_in_gaussian就可以不需要mask-in-gt-box，但是由于框包含旋转角，而mask-in-gt-box约束的是在不包含旋转角的水平框内部，因此实际的约束范围会变得更小一些，这也一定程度导致了即使有Gaussian约束，加上mask-in-gt-box约束后正样本的区域还是类似一个小区域的矩形)，不加mask-in-gt-box约束后，正样本的区域会和椭圆更类似，但是正样本区域也更大，过大的正样本区域可能也会带来一些副作用，因此center-百分比就是用来进一步限制椭圆区域的范围，当center-1时，椭圆区域正好内切于旋转矩形，百分比小于1时，椭圆区域就会越小，代表正样本区域越小。目前看来这个百分比不能太大也不能太小

more-pos表示原来回归分支仍进一步通过半径限制正样本范围(限制在正方形区域)，加了more-pos后回归分支则取消半径限制

cls_pos_cnt_all 表示分类分支只回归正样本，centerness分支回归所有样本

**upload to DOTA server evaluation result (testset):**

fcos_theta-weight1_adamw_lr1e-3_rotatediouloss_reg-centerness_trainval

|  PL   |  BD   |  BR   |  GTF  |  SV   |  LV   |  SH   |  TC   |  BC   |  ST   |  SBF  |  RA   |  HA   |  SP   |  HC   | <font color=Red>mAP50</font> | mAP75 |  mAP  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--------------------------: | :---: | :---: |
| 89.30 | 82.05 | 48.75 | 61.82 | 76.42 | 73.72 | 80.69 | 90.83 | 85.78 | 86.47 | 67.11 | 67.57 | 72.50 | 74.25 | 53.12 |          **74.03**           | 46.92 | 45.05 |

fcos_theta-weight1_adamw_lr1e-3_rotatediouloss_reg-centerness_gaussian-assigner_trainval

|  PL   |  BD   |  BR   |  GTF  |  SV   |  LV   |  SH   |  TC   |  BC   |  ST   |  SBF  |  RA   |  HA   |  SP   |  HC   | <font color=Red>mAP50</font> | mAP75 |  mAP  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--------------------------: | :---: | :---: |
| 89.62 | 84.27 | 48.81 | 68.53 | 77.46 | 74.47 | 81.29 | 90.79 | 87.11 | 86.41 | 67.07 | 64.64 | 73.18 | 77.83 | 58.07 |          **75.30**           | 47.80 | 45.74 |





## Visualize and Analysis

### GA的消融可视化

最大类别score激活热力图

![cat_score_map](https://github.com/Scienthusiasts/HeltonRotation/blob/dev/exp/cat_score_map.png)

centerness激活热力图

![centerness_map](https://github.com/Scienthusiasts/HeltonRotation/blob/dev/exp/centerness_map.png)

联合置信度(max_score*centerness)激活热力图

![joint_score_map](https://github.com/Scienthusiasts/HeltonRotation/blob/dev/exp/joint_score_map.png)

注释：

GA-w/o-mask-in-gtboxes-radious-1：采用高斯执行分配正负样本，去除mask-in-gtboxes的约束，高斯正样本的半径为1

GA-w/o-mask-in-gtboxes-radious-0.7：采用高斯执行分配正负样本，去除mask-in-gtboxes的约束，高斯正样本的半径为0.7

GA-w/o-mask-in-gtboxes-radious-0.7：采用高斯执行分配正负样本，去除mask-in-gtboxes的约束，高斯正样本的半径为0.5

GA：采用高斯执行分配正负样本，保留mask-in-gtboxes的约束，高斯正样本的半径为1

w/o-GA：采用最原始的FCOS正负样本分配策略

分析：

## Reference

[CAPTAIN-WHU/DOTA_devkit (github.com)](https://github.com/CAPTAIN-WHU/DOTA_devkit)

[YOLOv5_DOTAv1.5(遥感/无人机旋转目标检测，全踩坑记录) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/357992219)

[hukaixuan19970627/yolov5_obb: yolov5 + csl_label.(Oriented Object Detection)（Rotation Detection）（Rotated BBox）基于yolov5的旋转目标检测 (github.com)](https://github.com/hukaixuan19970627/yolov5_obb)

SCRDet: Towards More Robust Detection for Small, Cluttered and Rotated Objects (ICCV2019)

https://github.com/facias914/sood-mcl

