# 目前把其他方法引进这个框架的要求

- config文件中model字段下必须有backbone_name字段
- 自定义的数据集类必须叫做`DOTA2LongSideFormatYOLODataset`类
- `DOTA2LongSideFormatYOLODataset`类中必须dataset_collate和worker_init_fn静态方法
- 自定义的网络必须叫做Model类
- 自定义的网络类中必须包含`calcBatchLoss`方法，用于前向反向并计算损失，返回为损失的字典形式，其中必须包括字段`total_loss`
- 数据集.py文件中必须包含Transform类, 且有testTF成员

# 运行命令
CUDA_VISIBLE_DEVICES=1
python runner.py --config ./configs/yolov5.py

# 如何将YOLOv5改为旋转框检测
目前的主要修改逻辑是, 网络的box部分和角度部分的回归完全解耦开来, 独立进行回归和计算损失, anchor的形式也不用变
对于box回归部分,把box完全当成水平框去对待(直接就是旋转宽的wh,而不是其外接水平框的wh), box依然计算水平框的iou损失(和水平框完全一致不用更改), 不会用到skewIoU
且正负样本分配的基本逻辑也不用更改, 还是完全基于YOLOv5水平框的那一套逻辑
对于角度回归的部分, 目前先测试一下直接采用SmoothL1Loss进行回归能否收敛(会存在明显的边界问题), 之后再尝试其他的方法
[如何缓解角度的边界问题:从损失函数入手; 从旋转目标的定义入手,定义一种新的不存在便捷问题的有向目标表示方式]

- 修改数据集读取的方式:
  本项目支持基于DOTA格式的8参表示法和基于YOLO格式的5参长边表示法, 目前默认角度范围为[-180, 0), 
  对于DOTA格式，首先会使用opencv的最小外接旋转矩形的函数处理转化为五参表示法, 最后都会统一转为YOLO的长边格式
  自定义实现了旋转,翻转,rotatedMosaic等增强方式(因为Albumentation库好像不支持旋转框的增强), rotatedMosaic增强还有点小Bug,可能会出现超出图像之外的框消除不掉的情况
  此外, YOLOv5的正负样本分配策略是静态的, 这部分主要添加y_trues中角度参数的部分, 同时将角度GT归一化到0,1之间

- 修改网络部分:
  其实主要是Head部分,Backbone和Neck可以完全不动, Head主要针对添加角度回归这一参数进行修改
  网络结构部分,添加对角度的回归, 回归的是归一化到0,1的角度(目前是对回归的结果取sigmoid, 先看下能不能收敛)
  对于BatchLoss函数，主要添加角度回归损失

- 修改评估,测试部分的逻辑:
  测试和评估涉及到需要计算nms(训练的时候没用到NMS),因此对于这部分的修改,需要引入旋转IoU和旋转框的NMS,这部分我直接基于mmcv.ops中的rotated算子进行实现
  对于测试一张图像, 只需要将预测的theta作用到回归框的坐标上即可,这里主要修改可视化函数，添加rbox2PolyNP方法将5参表示法转化为8参表示法,再利用cv2.drawContours进行绘制
  对于评估整个验证集,这里遵循DOTA_devikit的评估方式, 即将模型的预测结果按照类别分别保存在相应类别的txt文档当中(格式为: file_name, score, 8个坐标), 评估时直接调用DOTA_devikit下dota_evaluation_task1.voc_eval方法

**update_24_7_27**

打算将角度回归部分与head部分解耦, 方便后续的修改与调整

目前Head每个通道的含义：cx, cy, w, h, θ, obj_score, cls_logits



# 如何在test上测试并提交DOTA服务器

目前的方法，以后可以修改整合

- 将config文件下的`val_img_dir`字段设置为test图像所在目录，然后进行测试，报错了不用管,因为test没有GT文件
- 在`utils\metrics.py`下执行merge对应方法即得到整合后的模型推理结果(将split上的推理结果合并)
- 在https://captain-whu.github.io/DOTA/evaluation.html提交