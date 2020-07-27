#!/usr/bin/env python
import torch
import cv2
import numpy as np
import argparse
import sys
import os
import shutil
from keypoint_net import KeypointNet
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from akhenaten_dv.msg import frame_msg
import rospy
import time

class KeyPointsDetection:
    def __init__(self):
        self.weights = "./src/akhenaten_dv/scripts/Perception/KPDetection/weights.pt"
        self.img_size = 512
        self.model = KeypointNet()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        model_dict = torch.load(self.weights).get('model')
        for k in model_dict.keys():
            model_dict[k] = model_dict[k].cuda()
        self.model.load_state_dict(model_dict)
        self.model.eval()    
        self.it = 0
        self.image_patches = []
    def detect(self, images):
        output = self.model(images)
        return output[1][0].cuda().data


kpDet = KeyPointsDetection()
bridge = CvBridge()
kps_pub = rospy.Publisher('Frame_BBox_Color_Kps', frame_msg)
def callback(data):
    img = bridge.imgmsg_to_cv2(data.image_frame)
    bboxes = data.BBox
    print("KP Detection: Recieved frame")
    st = time.time()
    kps = []
    for i in range(0,int(len(bboxes)),4):
        x = bboxes[i]
        y = bboxes[i+1]
        w = bboxes[i+2]
        h = bboxes[i+3] 
        cropped_img = img[y:y+h, x:x+w]
        cropped_img = cv2.resize(cropped_img, (80,80))
        # cv2.imshow("os patch", cropped_img)
        # cv2.waitKey(0)
        
        cropped_img = (cropped_img.transpose((2, 0, 1)) / 255.0)[np.newaxis, :]
        cropped_img = torch.from_numpy(cropped_img).type('torch.cuda.FloatTensor')
        dets = np.array(kpDet.detect(cropped_img).cpu())
        for i in range(len(dets)):
            kps.append(dets[i][0])
            kps.append(dets[i][1])
    # k=0
    # p=0
    # for i in range(0, int(len(kps)),14):
    #     x = bboxes[p]
    #     y = bboxes[p+1]
    #     w = bboxes[p+2]
    #     h = bboxes[p+3]
    #     p+=4
    #     for j in range(0,14,2):
    #         pt = [kps[i+j]*w+x, kps[i+j+1]*h+y]
    #         cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (255,0,0), -1)
    #         k=+1
    # print(len(kps))
        #kps.extend((.reshape(14).tolist()))

    # cv2.imshow("os", img)
    # cv2.waitKey(0)
      
    msg = frame_msg()
    msg.image_frame = data.image_frame
    msg.BBox = data.BBox
    msg.Color = data.Color
    msg.Kps = kps
    kps_pub.publish(msg)
    end = time.time()
    print("Keypoints detected in ",(end-st),' seconds')


if __name__ == '__main__':
    try:
        print("KP Detection: running @", os.getcwd())
        rospy.init_node('KP_det', anonymous=False)
        rospy.Subscriber('Frame_BBox_Color', frame_msg, callback)
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
