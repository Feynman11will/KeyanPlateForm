# coding: utf-8
# This script is modified from https://github.com/lars76/kmeans-anchor-boxes

from __future__ import division, print_function

import numpy as np
import argparse
import re
import os
def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)
    # iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    param:
        boxes: numpy array of shape (r, 4)
    return:
    numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)

def kmeans1(boxes,k,dist=np.mean):
    rows =  boxes.shape[0]
    distances = np.zeros([rows,k])
    np.random.seed()

    clusters = boxes[np.random.choice(rows, k, replace=False)]
    cur_clusters = np.zeors([k,2])
    last_clusters = np.zeors([rows,])
    while(True):

        for r in range(rows):
            distances[r] = 1 - iou(boxes[r],clusters)
        G = np.argmin(distances,axis = 1)
#         求均值
        if (last_clusters== G).all():
            break
        for k_i in range(k):
            cur_clusters[k_i] = dist(boxes[G==k_i],axis = 0)
        clusters  =cur_clusters
    return clusters
def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """

    rows = boxes.shape[0]
    # 没有初始化 ， 形状为【rors,k】
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    # 从 [0,rows-1]取出k个数字
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        # 按照iou计算距离最近、最小的作为簇
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break
        # 每一个类别取中位数
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

# 获取每一个边界框的场和宽，并进行归一化
def parse_anno1(annotation_path, lb_list,target_size=None):
    file_list = open(annotation_path, 'r')
    # list_file = [lt for lt in file_list]
    file2 = []
    res = []
    for file in file_list:
        lables = os.path.join(lb_list , file.replace("\n", "").strip().split('/')[-1].split('.')[0] + '.txt')
        lable_list = open(lables, 'r')
        for lable in lable_list:
            obj = lable.replace("\n", "").strip().split(' ')[1:]
            width = int(float(obj[2])*target_size[0])
            height = int(float(obj[3])*target_size[1])
            res.append([width,height])
    res = np.asarray(res)
    return res

def parse_anno(annotation_path, target_size=None):
    anno = open(annotation_path, 'r')
    result = []
    for line in anno:
        s = line.strip().split(' ')
        img_w = int(s[2])
        img_h = int(s[3])
        s = s[4:]
        box_cnt = len(s) // 5
        for i in range(box_cnt):
            x_min, y_min, x_max, y_max = float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3]), float(s[i*5+4])
            width = x_max - x_min
            height = y_max - y_min
            assert width > 0
            assert height > 0
            # use letterbox resize, i.e. keep the original aspect ratio
            # get k-means anchors on the resized target image size
            if target_size is not None:
                resize_ratio = min(target_size[0] / img_w, target_size[1] / img_h)
                width *= resize_ratio
                height *= resize_ratio
                result.append([width, height])
            # get k-means anchors on the original image size
            else:
                result.append([width, height])
    result = np.asarray(result)
    return result



def get_kmeans(anno, cluster_num=9):

    anchors = kmeans(anno, cluster_num)
    ave_iou = avg_iou(anno, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou

def kmeans_out(opt):
    parant_path = opt.ds_dir
    lb_path = os.path.join(parant_path, 'labellist')
    # / data1 / wanglonglong / FeiYan / labels
    lb_list = os.path.join(parant_path, 'labels')
    target_size = [opt.img_size, opt.img_size]
    annotation_path = os.path.join(lb_path ,'train.txt')

    anno_result = parse_anno1(annotation_path, lb_list, target_size=target_size)
    anchors, ave_iou = get_kmeans(anno_result, 9)

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{},'.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-1]

    print('anchors are:')
    print(anchor_string)
    print('the average iou is:')
    print(ave_iou)

    path = parant_path
    print(path)

    with open(os.path.join(path,'kmeans.txt'), 'w+') as file :
        file.write(anchor_string)
    

if __name__ == '__main__':
    # target resize format: [width, height]
    # if target_resize is speficied, the anchors are on the resized image scale
    # if target_resize is set to None, the anchors are on the original image scale
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-size', type=int, default=640)
    parser.add_argument('--parant-path', type=str, default='/data1/wanglonglong/FeiYan/',help='数据集的父路径')
    parser.add_argument('--lb-path', type=str, default='/data1/wanglonglong/FeiYan/labellist/',help='标签集列表的路径')
    parser.add_argument('--lb-list', type=str, default='/data1/wanglonglong/FeiYan/labels/', help='标签集文件的路径')
    parser.add_argument('--cfg-path', type=str, default='./cfg/yolov3-xray.cfg', help='配置文件路径')

    opt = parser.parse_args()
    parant_path = opt.parant_path
    lb_path = opt.lb_path
    lb_list = opt.lb_list

    target_size = [opt.target_size, opt.target_size]

    annotation_path = lb_path + 'train.txt'
    anno_result = parse_anno1(annotation_path, lb_list,target_size=target_size)

    anchors, ave_iou = get_kmeans(anno_result, 9)

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-2]

    print('anchors are:')
    print(anchor_string)
    print('the average iou is:')
    print(ave_iou)

    path = opt.cfg_path
    print(path)
    # file1 = open(path,'r')
    file = open(path, 'r')
    # print(file)
    lines = file.read().split('\n')
    tmp = ""

    for line in lines:
        line +='\n'
        if line.split('=')[0].strip() == 'anchors':
            line = 'anchors = ' + anchor_string  + '\n'
        tmp += line
    print('解析完成\n')
    print(tmp)
    file.close()
    file = open(path, 'w+')

    for line in tmp:
        file.write(line)
    file.close()
    print('输出完成\n')
    import os
    # os.rename("cfg/yolov3-xray2.txt", "cfg/yolov3-xray2.cfg")
