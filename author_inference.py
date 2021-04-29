# coding=utf-8
'''
@ Summary: 作者的推理代码, 校验用
@ Update:  

@ file:    author_inference.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2021/1/26 上午11:05
'''
import os
import cv2
import time
import numpy as np
from pathlib import Path
from tensorflow import keras
from common.yolo_postprocess_np import yolo_decode, yolo_handle_predictions, yolo_correct_boxes, yolo_adjust_boxes

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def inference(img_path, model):
    img_raw = cv2.imread(str(img_path))
    # 灰度图 resize
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    # shape: (160, 160)
    img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_LINEAR )
    # RGB 图 resize
    # image_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(image_rgb, (320, 320),
    #                            interpolation=cv2.INTER_LINEAR)
    img_copy = img.copy()

    # normalize
    img = img / 255.0
    img = np.asarray(img).astype('float32')

    # expand channel
    img = np.expand_dims(img, axis=-1)

    # expand batch --> (1, 160, 160, 1)
    input = np.expand_dims(img, axis=0)

    # shape: (1, 5, 5, 30)
    # pred = time.time()
    yolo_output = model.predict(input)
    # print(f"true inference time: {time.time()-pred} s...")

    return img_raw, yolo_output


def draw_img(boxes, img):
    height, weight = img.shape[:2]
    # show img
    x, y, w, h = boxes
    xmin = int((x - w / 2) * weight)
    xmax = int((x + w / 2) * weight)
    ymin = int((y - h / 2) * height)
    ymax = int((y + h / 2) * height)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.imshow('results', img)
    cv2.waitKey(2000)


def main():
    # person.jpg 57; 000001.jpg 63
    img_path = "./example/004.jpg"
    model_path = './weights/yolo-s.h5'
    root_path = Path("../../images/001")


    # load model
    model = keras.models.load_model(model_path)

    # 预选框
    anchor = [[13, 24], [33, 42], [36, 87], [94, 63], [68, 118]]

    # one image

    img, yolo_output = inference(img_path, model)
    img_shape = img.shape[:2]
    predictions = yolo_decode(yolo_output, anchor, num_classes=1,
                              input_dims=(160, 160), scale_x_y=0,
                              use_softmax=False)
    predictions[..., :2] *= img_shape[::-1]
    predictions[..., 2:4] *= img_shape[::-1]
    # xy --> left, top
    predictions[..., :2] -= (predictions[..., 2:4] / 2)
    boxes, classes, scores = yolo_handle_predictions(predictions, img_shape)
    print(boxes)
    print(classes)
    print(scores)
    # draw_img(boxes, img)



if __name__ == "__main__":
    main()