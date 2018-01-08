# -*- coding: utf-8 -*-
# !/usr/bin/env python

# @file: huanlepintu.py.py
# @Author: Molin
# @Contact: 891680996@qq.com
# @Date: 1/6/2018
# @Last Modified Time: 9:23 PM

import cv2 as cv
import numpy as np
import os
from operator import itemgetter
import sys
# img_x = cv.imread("start.jpg")
# img_start = cv.resize(img_x, (540, 960), interpolation=cv.INTER_CUBIC)
# # cv.imshow("img", img_start)
# # cv.waitKey()
# img_y = cv.imread("end.jpg")
# img_end = cv.resize(img_y, (540, 960), interpolation=cv.INTER_CUBIC)
# gray_start = cv.cvtColor(img_start, cv.COLOR_BGR2GRAY)
# gray_end = cv.cvtColor(img_end, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(gray_start, 100, 255, cv.THRESH_BINARY)

global point
point = [(0, 0), (0, 0)]


def get_pic(img_x, img_y):
    img_x1 = cv.imread(img_x)
    xh, xw, xd= img_x1.shape
    # print(xh, xw)
    xw = int((768/xh)*xw)
    # print(xh, xw)
    img_start = cv.resize(img_x1, (xw, 768), interpolation=cv.INTER_AREA)
    img_y1 = cv.imread(img_y)
    # yh, yw, yd = img_y1.shape
    # print(yh, yw)
    # yw = int((768 / yh) * yw)
    # print(yh, yw)
    # 默认两张图片大小一样
    img_end = cv.resize(img_y1, (xw, 768), interpolation=cv.INTER_AREA)
    return img_start, img_end

def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        point.pop(0)
        point.append((x, y))
        print('point' + ':', point[0], point[1])


def img_copy(p, img):
    w0 = p[0][0]
    w1 = p[1][0]
    h0 = p[0][1]
    h1 = p[1][1]
    # cv.imshow(np.copy(img[w0:w1, h0:h1]))
    return img[h0:h1, w0:w1]


def sort_result(l, n):
    result_index = []
    for i in range(1, (n + 1)):
        for j in range(1, (n + 1)):
            result_index.append((i, j))
    # print(result_index)
    for i in range(len(l) - 1):
        min_index = i
        for j in range(i + 1, len(l)):
            if (l[min_index] > l[j]):
                min_index = j
        l[i], l[min_index] = l[min_index], l[i]
        print('第 %2d 次循环' % (i + 1), '：', result_index[i], '与', result_index[min_index], '互换')


def main(img_x, img_y, n):
    cv.namedWindow('img_start')
    cv.setMouseCallback('img_start', on_mouse)
    img_start, img_end = get_pic(img_x, img_y)
    # gray_end = cv.cvtColor(img_end, cv.COLOR_BGR2GRAY)
    cv.imshow("img_start", img_start)
    cv.waitKey()
    p = point
    img_temp = img_copy(p, img_start)
    # img_temp = np.copy(img_start[p[0][1]:p[1][1],p[0][0]:p[1][0]])
    # cv.imshow("img_temp", img_temp)
    img_temp_gray = cv.cvtColor(img_temp, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img_temp_gray, 100, 255, cv.THRESH_BINARY)

    image, contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv.boundingRect(contours[-1])
    x, y, w, h = (x + 2, y + 2, w - 2, h - 2)
    print(w, h)
    result = []
    q = 0
    for i in range(0, n):
        for j in range(0, n):
            q = q + 1
            img = img_temp[(y + round(h * i / n)):(y + round(h * (i + 1) / n)),
                  (x + round(w * j / n)):(x + round(w * (j + 1) / n))]
            res = cv.matchTemplate(img, img_end, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            text_loc = (max_loc[0], max_loc[1] + 20)
            result.append((q, max_loc[0], max_loc[1]))
            font = cv.FONT_HERSHEY_SIMPLEX  # 使用默认字体
            img = cv.putText(img_end, str(i + 1) + str(j + 1), text_loc, font,
                             0.8, (255, 255, 255), 2)
    # print("000", result)
    result = sorted(result, key=itemgetter(2, 1))
    # print("111", result)
    for i in range(n):
        result[i * n:(i + 1) * n] = sorted(result[i * n:(i + 1) * n], key=itemgetter(1))
    # print("222", result)
    result_sort = [x[0] for x in result]
    # print("333", result_sort)
    sort_result(result_sort, n)
    cv.imshow("img", img)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python xxx.py start_pic end_pic n")
        exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
# else:img_x = "start.jpg", img_y = "end.jpg", n = 5
#     main("start.jpg", "end_jpg", 5)
