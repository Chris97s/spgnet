import numpy as np
import glob
import sys
import os
sys.path.append("/home/songzhx6/space_sda1/projects/SPGNet")
from utils.read_write_pts import read_pts_file
from scipy.spatial import procrustes
from Active_shape_model import *

def load_landmarks(dir_landmarks):
    
    inputNames = glob.glob(dir_landmarks)
    inputList = np.empty([len(inputNames), 65, 2])
    i = 0
    idlist = np.empty([len(inputNames)])
    # 使用切片提取 'a' 和 'b' 之间的内容
    for inputName in inputNames:
        # print(inputName)
        file_name = os.path.basename(inputName)
        parts = file_name.split('.pts')
        id = parts[0]
        points,num_pts,width,height = read_pts_file(inputName)
        points.append([width,height])
        print(id)
        print(len(points))
        inputList[i] = points
        idlist[i] =  "".join(list(filter(str.isdigit, id)))
        i+=1
        # print(inputList.shape)
    idlist = idlist.astype(int)
    return inputList,idlist

def total_procrustes_analysis(all_landmarks):
    #allign shapes in their set
    all_landmarks_std = np.empty_like(all_landmarks)
    mean = np.mean(all_landmarks,0)
    mean = mean.reshape(-1,2)
    for i, landmark in enumerate(all_landmarks):
        landmark = landmark.reshape(-1,2)
        mean_std, landmark_std, disp = procrustes(mean, landmark)
        landmark_std = landmark_std.reshape(-1)
        all_landmarks_std[i] = landmark_std

    return all_landmarks_std

def custom_distance(point1, point2):
    # 计算点之间的距离
    point1 = point1.reshape(-1,2)
    point2 = point2.reshape(-1,2)
    mtx1, mtx2, disparity = procrustes(point1, point2)
    return disparity