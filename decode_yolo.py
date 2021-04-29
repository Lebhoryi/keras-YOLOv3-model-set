# coding=utf-8
'''
@ Summary: valid yolo-s model
@ Update:  

@ file:    inference_yolo-s.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2021/1/19 下午4:10

@ Update： 1. 置信度计算方式更改，之前是(x,y,w,h,scores,class) 中scores 直接拿来用，
               发现当没有检测出物体的时候，也会给一个错误的box，现改为scores*class
           2. 置信度增加0.1阈值，过滤掉没有物体检测出来的时候的错误的box
@ Date:    2021/1/22

@ Update:  新增nms; 取消最大值选框
@ Date:    2021/1/26
'''

import cv2
from pathlib import Path
import numpy as np
from tensorflow import keras

def inference(img_path, model):
    img_raw = cv2.imread(img_path)
    # 灰度图 resize
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    # shape: (160, 160)
    img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_LINEAR )

    # normalize
    img = img / 255.0
    img = np.asarray(img).astype('float32')

    # expand channel
    img = np.expand_dims(img, axis=-1)

    # expand batch --> (1, 160, 160, 1)
    input = np.expand_dims(img, axis=0)

    # shape: (1, 5, 5, 30)
    yolo_output = model.predict(input)

    return img_raw, yolo_output


def draw_img(boxes, img):
    # show img
    x, y, w, h = boxes
    xmin = int(x - w / 2)
    xmax = int(x + w / 2)
    ymin = int(y - h / 2)
    ymax = int(y + h / 2)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.imshow('results', img)
    cv2.waitKey(2000)  # 2s 之后关闭


def yolo_decode(prediction, anchors, num_classes, input_dims, scale_x_y=None, use_softmax=False):
    '''Decode final layer features to bounding box parameters.'''
    num_anchors = len(anchors)  # anchor 的数量
    grid_size = prediction.shape[1:3]  # 将一张图片分割成5*5

    # shape: (125, 6)
    prediction = np.reshape(prediction,
                            (grid_size[0] * grid_size[1] * num_anchors, num_classes + 5))

    # generate x_y_offset grid map
    x_y_offset = [[[j, i]] * grid_size[0] for i in range(grid_size[0]) for j in range(grid_size[0])]
    x_y_offset = np.array(x_y_offset).reshape(grid_size[0] * grid_size[1] * num_anchors , 2)

    x_y_tmp = 1 / (1 + np.exp(-prediction[..., :2]))
    box_xy = (x_y_tmp + x_y_offset) / np.array(grid_size)[::-1]

    # Log space transform of the height and width
    anchors = np.array(anchors*(grid_size[0] * grid_size[1]))
    box_wh = (np.exp(prediction[..., 2:4]) * anchors) / np.array(input_dims)[::-1]

    # sigmoid function
    objectness = 1 / (1 + np.exp(-prediction[..., 4:5]))

    # sigmoid function
    if use_softmax:
        class_scores = np.exp(prediction[..., 5:]) / np.sum(np.exp(prediction[..., 5:]))
    else:
        class_scores = 1 / (1 + np.exp(-prediction[..., 5:]))

    return np.concatenate((box_xy, box_wh), axis=-1), objectness, class_scores


def non_max_suppress(boxes, classes, scores, threshold=0.4):
    """ 简洁版 hard nms 实现, 单类别"""

    # center_xy, box_wh
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    # left, top, right, bottom
    left = x - w / 2
    top = y - h / 2
    right = x + w / 2
    bottom = y + h / 2
    # sorted index, 从大到小
    order = np.argsort(scores)[::-1]
    ares = w * h

    keep = []  # 保留的有效的索引值, list
    # 保留置信度最高的box, 其余依次与其遍历, 删除大于阈值的box,剩
    # 下的继续保留最高置信度的box, 依次迭代
    while order.size > 0:
        keep.append(order[0])  # 永远保留置信度最高的索引
        # 最大置信度的左上角坐标分别与剩余所有的框的左上角坐标进行比较，分别保存较大值
        inter_xmin = np.maximum(left[order[0]], left[order[1:]])
        inter_ymin = np.maximum(top[order[0]], top[order[1:]])
        inter_xmax = np.minimum(right[order[0]], right[order[1:]])
        inter_ymax = np.minimum(bottom[order[0]], bottom[order[1:]])

        # 当前类所有框的面积
        # x1=3,x2=5,习惯上计算x方向长度就是x=3、4、5这三个像素，即5-3+1=3，
        # 而不是5-3=2，所以需要加1
        inter_w = np.maximum(0., inter_xmax - inter_xmin + 1)
        inter_h = np.maximum(0., inter_ymax - inter_ymin + 1)
        inter = inter_w * inter_h

        #计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        iou = inter / (ares[order[0]] + ares[order[1:]] - inter)

        # 计算iou的时候, 并没有计算第一个数, 所以索引对应的是order[1:]之后的, 所以需要加1
        indexs = np.where(iou <= threshold)[0]
        order = order[indexs+1]

    keep_boxes = boxes[keep]
    keep_classes = classes[keep]
    keep_scores = scores[keep]

    return keep_boxes, keep_classes, keep_scores


def filter(pred_xywh, objectness, class_scores, img_shape, iou_threshold=0.4, confidence=0.1):
    """ 得到真正的置信度，并且过滤 """
    # shape: (125, 1)
    box_scores = objectness * class_scores
    assert box_scores.shape[-1] == 1, "有不止一个类别, 该函数不可用, 仅对单类别使用"

    box_scores = np.squeeze(box_scores, axis=-1)

    # filter
    pos = np.where(box_scores >= confidence)

    if not pos:
        print("No person detected!!!")
        return
    # get all scores and boxes
    scores = box_scores[pos]
    boxes = pred_xywh[pos]
    classes = np.zeros(scores.shape, dtype=np.int8)  # 单类别

    # 相对坐标转为真实坐标
    boxes[..., :2] *= img_shape
    boxes[..., 2:] *= img_shape

    nboxes, nclasses, nscores = non_max_suppress(boxes, classes, scores, threshold=iou_threshold)

    return nboxes, nclasses, nscores


def main():
    # person.jpg 57; 000001.jpg 63
    img_path = "./example/004.jpg"
    model_path = './weights/yolo-s.h5'

    # load model
    model = keras.models.load_model(model_path)

    # yolo_output, shape: (1, 5, 5, 30)
    img, yolo_output = inference(str(img_path), model)
    img_shape = img.shape[:-1][::-1]  # weight, height

    # 预选框
    anchor = [[13, 24], [33, 42], [36, 87], [94, 63], [68, 118]]

    pred_xywh, objectness, class_scores = yolo_decode(yolo_output, anchor, num_classes=1,
                              input_dims=(160, 160), scale_x_y=0,
                              use_softmax=False)
    boxes, classes, scores = filter(pred_xywh, objectness, class_scores, img_shape)
    for box in boxes:
        draw_img(box, img)


if __name__ == "__main__":
    main()