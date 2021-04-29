# coding=utf-8
'''
@ Summary: valid yolo-s model
@ Update:  

@ file:    inference_yolo-s_bak.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2021/1/19 下午4:10
'''
import cv2
import numpy as np
from tensorflow import keras
from common.yolo_postprocess_np import yolo_decode, yolo_handle_predictions, yolo_correct_boxes, yolo_adjust_boxes


def letterbox_image(img, inp_dim=(160, 160)):
    """
    lteerbox_image()将图片按照纵横比进行缩放，将空白部分用(128,128,128)填充,调整图像尺寸
    具体而言,此时某个边正好可以等于目标长度,另一边小于等于目标长度
    将缩放后的数据拷贝到画布中心,返回完成缩放
    """
    img_h, img_w,  = img.shape[:2]
    w, h = inp_dim#inp_dim是需要resize的尺寸（如416*416）
    # 取min(w/img_w, h/img_h)这个比例来缩放，缩放后的尺寸为new_w, new_h,即保证较长的边缩放后正好等于目标长度(需要的尺寸)，另一边的尺寸缩放后还没有填充满.
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC) #将图片按照纵横比不变来缩放为new_w x new_h，768 x 576的图片缩放成416x312.,用了双三次插值
    # 创建一个画布, 将resized_image数据拷贝到画布中心。
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128, dtype=np.int8)#生成一个我们最终需要的图片尺寸hxwx3的array,这里生成416x416x3的array,每个元素值为128
    # 将wxhx3的array中对应new_wxnew_hx3的部分(这两个部分的中心应该对齐)赋值为刚刚由原图缩放得到的数组,得到最终缩放后图片
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_image

    cv2.imshow('result', canvas)
    cv2.waitKey(2000)
    return canvas


def yolo_decode2(prediction, anchors, num_classes, input_dims, scale_x_y=None, use_softmax=False):
    '''Decode final layer features to bounding box parameters.'''
    prediction = np.squeeze(prediction, axis=0)
    num_anchors = len(anchors)  # anchor 的数量

    grid_size = prediction.shape[0:2]  # 将一张图片分割成5*5 w, h
    #check if stride on height & width are same
    assert input_dims[0]//grid_size[0] == input_dims[1]//grid_size[1], 'model stride mismatch.'
    # stride = input_dims[0] // grid_size[0]  # 160/5 = 32 一共要进行32次操作

    # shape: (125, 6)
    prediction = np.reshape(prediction,
                            (grid_size[0] * grid_size[1] * num_anchors, num_classes + 5))

    # generate x_y_offset grid map
    x_y_offset = [[[j, i]] * grid_size[0] for i in range(grid_size[0]) for j in range(grid_size[0])]
    x_y_offset = np.array(x_y_offset).reshape(grid_size[0] * grid_size[1] * num_anchors , 2)

    x_y_tmp = 1 / (1 + np.exp(-prediction[..., :2]))
    x_y_tmp = x_y_tmp * scale_x_y - (scale_x_y - 1) / 2 if scale_x_y else x_y_tmp
    box_xy = (x_y_tmp + x_y_offset) / np.array(grid_size)[::-1]

    # Log space transform of the height and width
    anchors = np.array(anchors*(grid_size[0] * grid_size[1]))
    box_wh = (np.exp(prediction[..., 2:4]) * anchors) / np.array(input_dims)[::-1]

    # sigmoid function
    scores = 1 / (1 + np.exp(-prediction[..., 4]))

    # sigmoid function
    classes = 1 / (1 + np.exp(-prediction[..., 5:]))

    return np.concatenate((box_xy, box_wh), axis=-1), scores, classes


def yolo_decode3(prediction, anchors, num_classes, input_dims, scale_x_y=None, use_softmax=False):
    """ 先计算置信度最高的索引，然后计算xywh class"""
    prediction = np.squeeze(prediction, axis=0)
    num_anchors = len(anchors)  # anchor 的数量 5
    grid_size = prediction.shape[0:2]  # 将一张图片分割成 5*5
    # shape: (125, 6)
    prediction = np.reshape(prediction,
                  (grid_size[0] * grid_size[1] * num_anchors, num_classes + 5))

    # sigmoid function
    scores = 1 / (1 + np.exp(-prediction[..., 4]))
    max_score_index = np.argmax(scores)

    # 求 (cx, cy), 每一个g rid 的左上角坐标
    grid_index = max_score_index // num_anchors + 1
    cx = grid_index // grid_size[0]
    cy = grid_index % grid_size[1] - 1

    # 求 box_xy
    # sigmoid function
    x_y_tmp = 1 / (1 + np.exp(-prediction[..., :2]))
    x_y_tmp = x_y_tmp * scale_x_y - (scale_x_y - 1) / 2 if scale_x_y else x_y_tmp
    box_xy = (x_y_tmp[max_score_index, ...] + np.array([cy, cx])) / np.array(grid_size)[::-1]

    # 求 box_wh
    anchor_index = max_score_index % num_anchors
    assert anchor_index <= num_anchors, "cal anchor index wrong!!!"
    box_wh = (np.exp(prediction[..., 2:4][max_score_index, ...]) * anchors[anchor_index]) / np.array(input_dims)[::-1]

    return np.concatenate((box_xy, box_wh), axis=-1), scores[max_score_index]


def draw_img(pred_xywh, raw_img):
    height, weight = raw_img.shape
    # show img
    x, y, w, h = pred_xywh
    xmin = int((x - w / 2) * weight)
    xmax = int((x + w / 2) * weight)
    ymin = int((y - h / 2) * height)
    ymax = int((y + h / 2) * height)
    cv2.rectangle(raw_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.imshow('results', raw_img)
    cv2.waitKey(2000)


def gt_yolo_detection(yolo_output, anchor, num_classes, image_shape, img):
    # gt yolo_decode
    predictions = yolo_decode(yolo_output, anchor, num_classes,
                              input_dims=(160,160), scale_x_y=1.05,
                              use_softmax=True)
    predictions = yolo_correct_boxes(predictions, image_shape, model_image_size=(160,160))

    predictions = np.squeeze(predictions, axis=0)

    index = np.argmax(predictions[..., 4])
    left, top, weight, height = [predictions[..., i][index] for i in range(4)]
    right, bottom = left + weight, top + height

    cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
    cv2.imshow("results", img)
    cv2.waitKey(1000)

    return predictions[..., :4], predictions[..., 4], predictions[..., 5:]


def main():
    raw_img = cv2.imread("./example/person.jpg")
    height, weight, _ = raw_img.shape
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_LINEAR )
    img_copy = img.copy()

    # normalize
    img = img / 255.0
    img = np.asarray(img).astype('float32')

    # expand channel
    img = np.expand_dims(img, axis=-1)

    # expand batch --> (1, 160, 160, 1)
    input = np.expand_dims(img, axis=0)

    # print(f"img shape: {img.shape}")

    # load model
    model = keras.models.load_model('./weights/yolo-s.h5')
    # print(model.summary())
    # shape: (1, 5, 5, 30)
    yolo_output = model.predict(input)

    anchor = [[13, 24], [33, 42], [36, 87], [94, 63], [68, 118]]
    num_classes = 1
    # shape: (5, 5, 30)
    gt = False
    if gt:
        pred_xywh, pred_conf, pred_prob = gt_yolo_detection(yolo_output, anchor, num_classes,
                                                            image_shape=(height, weight), img=raw_img)
    else:
        pred_xywh, pred_conf, pred_prob = yolo_decode2(yolo_output, anchor, num_classes,
                                   input_dims=(160,160), scale_x_y=1.05,
                                   use_softmax=True)
        # draw img
        draw_img(pred_xywh[np.argmax(pred_conf)], img_copy)

    print(f"the prediction max score index: {np.argmax(pred_conf)}")
    print(f"the prediction score: {pred_conf[np.argmax(pred_conf)]}")
    print(f"the prediction box: {pred_xywh[np.argmax(pred_conf)]}")



    # # 节省计算量的方法
    # pred_xywh2, pred_conf2 = yolo_decode3(yolo_output, anchor, num_classes,
    #                                    input_dims=(160,160), scale_x_y=1.05,
    #                                    use_softmax=True)
    #
    # draw_img(pred_xywh2, weight, height, raw_img)
    # # print(pred_xywh2)
    # # print(pred_conf2)
    # if (pred_xywh2 == pred_xywh[np.argmax(pred_conf)]).all() \
    #         and pred_conf2 == float(pred_conf[np.argmax(pred_conf)]):
    #     print("yeah, u a right!")
    # else:
    #     print("Oh, u a wrong!!!")



if __name__ == "__main__":
    main()