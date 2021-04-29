# coding=utf-8
'''
@ Summary: 得到不同尺寸的图片, 用来检测yolo-s能识别的最小人形生物占比
@ Update:  

@ file:    get_diff_shape_image.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2021/1/28 下午4:07
'''
import cv2
import numpy as np
from pathlib import Path

def main():
    img_path = "./example/person.jpg"
    img = cv2.imread(img_path)
    height, weight = img.shape[:2]
    scale_lists = list(range(30, 70))
    scales = np.array(scale_lists) / 100
    new_heights, new_weights = height * scales, weight * scales
    for i in range(40):
        # resize
        resize_img = cv2.resize(img, (int(new_weights[i]), int(new_heights[i])))
        padding_w = int((weight - int(new_weights[i])) / 2)
        padding_h = int((height - int(new_heights[i])) / 2)
        full_img = np.pad(resize_img, ((padding_h, padding_h), (padding_w, padding_w), (0, 0)),
                          'constant', constant_values=0)
        scale_img_name = Path("./example/person") / ('person_' + str(scale_lists[i]) + '.jpg')
        cv2.imwrite(str(scale_img_name), full_img)

        cv2.imshow("resize", full_img)
        cv2.waitKey(500)
    # print(scale_lists)


if __name__ == "__main__":
    main()