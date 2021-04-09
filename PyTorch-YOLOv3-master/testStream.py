from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
from util import *
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import cv2
import numpy as np
import time
font =  cv2.FONT_HERSHEY_SIMPLEX
cap=cv2.VideoCapture(0)
start = time.time()
i =0
fps = 0
device = "cpu"
inp_dim = 416
"""model = Darknet("config/yolov3-tiny.cfg", inp_dim).to(device)
weights_path = "weights/yolov3-tiny.weights"
model.load_darknet_weights(weights_path)
model.eval()
class_path="data/coco.names"
num_classes = 80
classes = load_classes(class_path)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index
confidence = float(0.5)
nms_thesh = float(0.1)
def prep_image(img, inp_dim):
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim"""


while True:
    ret, frame= cap.read()
    """img = prep_image(frame, inp_dim)
    
    # Time elapsed
    
    with torch.no_grad():
            output = model(Variable(img[0].type(Tensor)))
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)
            #detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)"""
    end = time.time()
    seconds = end - start
    print ("Time taken : {0} seconds".format(seconds))
    i=i+1
    # Calculate frames per second
    if(i>10):
        fps  = 1 / seconds
        i=0
    print("Estimated frames per second : {0}".format(round(fps,2)))
    cv2.putText(frame,"{0} FPS".format(round(fps,2)),(8,20), font,0.8,(0,255, 0),2,cv2.LINE_AA)
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)
    start = time.time()
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()








