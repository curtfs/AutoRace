#!/usr/bin/env python
import os
import argparse
from os.path import isfile, join
import random
import tempfile
import time
import copy
import multiprocessing
import subprocess
import shutil
import cv2
import torch
import torch.nn as nn
from PIL import Image as img_pil
import rospy
from sensor_msgs.msg import Image
from akhenaten_dv.msg import frame_msg
import torchvision 
from models import Darknet
from utils.nms import nms
from utils.utils import xywh2xyxy, calculate_padding
import warnings
from tqdm import tqdm
from cv_bridge import CvBridge, CvBridgeError
warnings.filterwarnings("ignore")
import sys

class BBDetection():
    def __init__(self):
        print(os.getcwd())
        self.model_cfg = "./src/akhenaten_dv/scripts/Perception/BBoxDetection/model_cfg/yolo_baseline_tiny.cfg"
        self.weights_path = './src/akhenaten_dv/scripts/Perception/BBoxDetection/7.weights'
        self.conf_thres=0.8
        self.nms_thres=0.25
        self.vanilla_anchor = False
        self.xy_loss=2
        self.wh_loss=1.6
        self.no_object_loss=25
        self.object_loss=0.1
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        random.seed(0)
        torch.manual_seed(0)
        if cuda:
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        self.model = Darknet(config_path=self.model_cfg,xy_loss=self.xy_loss,wh_loss=self.wh_loss,no_object_loss=self.no_object_loss,object_loss=self.object_loss,vanilla_anchor=self.vanilla_anchor)
        # Load weights
        self.model.load_weights(self.weights_path, self.model.get_start_weight_dim())
        self.model.to(self.device, non_blocking=True)

    def detect(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img = img_pil.fromarray(cv_img)
        w, h = img.size
        new_width, new_height = self.model.img_size()
        pad_h, pad_w, ratio = calculate_padding(h, w, new_height, new_width)
        img = torchvision.transforms.functional.pad(img, padding=(pad_w, pad_h, pad_w, pad_h), fill=(127, 127, 127), padding_mode="constant")
        img = torchvision.transforms.functional.resize(img, (new_height, new_width))

        bw = self.model.get_bw()
        if bw:
            img = torchvision.transforms.functional.to_grayscale(img, num_output_channels=1)

        img = torchvision.transforms.functional.to_tensor(img)
        img = img.unsqueeze(0)
        
        with torch.no_grad():
            self.model.eval()
            img = img.to(self.device, non_blocking=True)
            # output,first_layer,second_layer,third_layer = model(img)
            output = self.model(img)


            for detections in output:
                detections = detections[detections[:, 4] > self.conf_thres]
                box_corner = torch.zeros((detections.shape[0], 4), device=detections.device)
                xy = detections[:, 0:2]
                wh = detections[:, 2:4] / 2
                box_corner[:, 0:2] = xy - wh
                box_corner[:, 2:4] = xy + wh
                probabilities = detections[:, 4]
                nms_indices = nms(box_corner, probabilities, self.nms_thres)
                main_box_corner = box_corner[nms_indices]
                if nms_indices.shape[0] == 0:
                    continue
            bboxes = []
            for i in range(len(main_box_corner)):
                x0 = main_box_corner[i, 0].to('cpu').item() / ratio - pad_w
                y0 = main_box_corner[i, 1].to('cpu').item() / ratio - pad_h
                x1 = main_box_corner[i, 2].to('cpu').item() / ratio - pad_w
                y1 = main_box_corner[i, 3].to('cpu').item() / ratio - pad_h 
                bboxes.append([x0,y0,x1,y1])

        return bboxes

bboxes_det = BBDetection()
img_patch = rospy.Publisher('Frame_BBox', frame_msg)
bridge = CvBridge()
def callback(data):
    frame_msg_send = frame_msg()
    frame_msg_send.image_frame = data
    print("BBox Detection: Recieved frame")
    st = time.time()
    img = bridge.imgmsg_to_cv2(data)
    bboxes = bboxes_det.detect(img)
    end = time.time()
    
    
    frame_bboxes=[]
    print("BBox Detection: BBoxes detected in ", (end -st), ' seconds')
    for i in range(len(bboxes)):
        h=int(bboxes[i][3]-bboxes[i][1])                         #[x0,y0,x1,y1]
        w=int(bboxes[i][2]-bboxes[i][0])
        x=int(bboxes[i][0])
        y=int(bboxes[i][1])
        h = h if h>0 else 0
        w = w if w>0 else 0
        x = x if x>0 else 0
        y = y if y>0 else 0
        frame_bboxes.append(x)
        frame_bboxes.append(y)
        frame_bboxes.append(w)
        frame_bboxes.append(h)
        cropped_img = img[y:y+h, x:x+w]
        # cv2.imshow("os patch", cropped_img)
        # cv2.waitKey(0)
        

    frame_msg_send.BBox = frame_bboxes
    frame_msg_send.Color = []
    img_patch.publish(frame_msg_send)
    end1 = time.time()
    print("BBox Detection: Published Frame Patches in ", (end1 - end), ' seconds')


if __name__ == '__main__':
    try:
        print("BBox Detection: running @", os.getcwd())
        rospy.init_node('bbox_det', anonymous=False)
        rospy.Subscriber('input_frames', Image, callback)
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
