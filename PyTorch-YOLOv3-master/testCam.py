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
import pickle as pkl
font =  cv2.FONT_HERSHEY_SIMPLEX
cap=cv2.VideoCapture(0)
start = time.time()
global k
k=0
fps = 0
device = "cpu"
inp_dim = 416
model = Darknet("config/yolov3-tiny.cfg", inp_dim).to(device)
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
colors = pkl.load(open("palette", "rb"))
abs_time = time.time()
def prep_image(img, inp_dim):
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    labeling = "{0}".format(classes[cls]) + " " + str(round(x[6].item() * 100, 2)) + " %"
    conf = str(x[6])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(labeling, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 1.5, c1[1] + t_size[1] + 2
    #cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, labeling, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 0.7, [225, 255, 255], 1);
    return img
    
while True:
    start_time = time.time()
    ret, frame= cap.read()
    img = prep_image(frame, inp_dim)
    co_im = np.asanyarray(frame)
    # Time elapsed
    k=k+1
    #with torch.no_grad():
    output = model(Variable(img[0].type(Tensor)))
    output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)
    output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim
    output[:, [1, 3]] *= frame.shape[1]
    output[:, [2, 4]] *= frame.shape[0]
    if(output.size()[0]>0):
        for i in range(output.size()[0]):
            w = output[i][1]+(output[i][3]-output[i][1])/2
            h = output[i][2]+(output[i][4]-output[i][2])/2
            #print("{0}  at".format(classes[int(output[i][-1])]))
            list(map(lambda x: write(x, co_im), output))
    d = round(1 / (time.time() - start_time), 2)
    if(k>10):
    	fps=d
    	k=0
    cv2.putText(frame,"{0} FPS".format(round(fps,2)),(8,20), font,0.8,(0,255, 0),2,cv2.LINE_AA)
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)
    start = time.time()
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

