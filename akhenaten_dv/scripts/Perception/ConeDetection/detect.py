#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import rospy
from sensor_msgs.msg import Image
from akhenaten_dv.msg import frame_msg
import math
from cv_bridge import CvBridge, CvBridgeError

class ConeDetection:
    def __init__(self):
        self.cfg = 'cfg/yolov3-spp.cfg'
        self.names='data/cones.names'
        self.weights='./src/akhenaten_dv/scripts/Perception/ConeDetection/last.pt'
        self.img_size = 320
        self.conf_thres = 0.3
        self.iou_thres=0.3
        self.fourcc = 'mp4v'
        self.half = False
        self.device=''
        self.view_img=False
        self.save_txt=False
        self.classes = None
        self.agnostic_nms=False
        self.augment=False
        self.cfg = list(glob.iglob('./**/' + self.cfg, recursive=True))[0]  # find file
        self.names = list(glob.iglob('./**/' + self.names, recursive=True))[0]  # find file
        # Initialize
        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else self.device)
        # Initialize model
        self.model = Darknet(self.cfg, self.img_size)
        # Load weights
        #attempt_download(weights)
        if self.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(self.model, self.weights)
        # Eval mode
        self.model.to(self.device).eval()
        self.names = load_classes(self.names)

    def detect(self, img0):
        model = self.model
        device = self.device
        cones_boxes = []
        cones_classes = []
        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = model(img, augment=self.augment)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                multi_label=False, classes=self.classes, agnostic=self.agnostic_nms)
        # Process detections
        cones_boxes = []
        cones_classes = []
        for i, det in enumerate(pred):  # detections for image i
            s, im0 = '', img0
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    cones_boxes.append(xywh)
                    cones_classes.append(int(cls))


        
        return [cones_boxes, cones_classes]

    def findColor(self, bbox, dets, img0):
        i = None
        colorbboxes = dets[0]
        cost = 0
        h2=int(bbox[3])                         #[x0,y0,x1,y1]
        w2=int(bbox[2])
        x2=int(bbox[0])
        y2=int(bbox[1])
        for j in range(len(colorbboxes)):
            h1=int(colorbboxes[j][3]*img0.shape[0])
            w1=int(colorbboxes[j][2]*img0.shape[1])
            x1=int(colorbboxes[j][0]*img0.shape[1] - w1/2)
            y1=int(colorbboxes[j][1]*img0.shape[0] - h1/2)
            
            cost = abs(h1-h2)+abs(w1-w2)+abs(x1-x2)+abs(y1-y2)
            if cost < 20:
                i = j
                break

        return dets[1][i] if i is not None else 3
    
    def setup_bboxes(bboxes_corrupt):
        BBoxes=[]
        for i in range(0, len(bboxes_corrupt), 4):
            BBoxes.append([bboxes_corrupt[i],bboxes_corrupt[i+1],bboxes_corrupt[i+2],bboxes_corrupt[i+3]])
        return BBoxes

coneDet = ConeDetection()
bridge = CvBridge()
colors_publisher = rospy.Publisher('Frame_BBox_Color', frame_msg)
Colors = []
BBoxes = []
def callback(data):
    frame_bbox_colors = frame_msg()
    frame_bbox_colors.image_frame = data.image_frame
    frame_bbox_colors.BBox = data.BBox
    frame_bbox_colors.Color = []
    BBoxes = ConeDetection.setup_bboxes(data.BBox)
    print("Color Detection: Recieved frame")
    st = time.time()
    img = bridge.imgmsg_to_cv2(data.image_frame)
    dets = coneDet.detect(img)
    end = time.time()
    print("Color detected in ", (end-st), ' seconds')
    for box in BBoxes:
        frame_bbox_colors.Color.append(coneDet.findColor(box, dets, img))

    colors_publisher.publish(frame_bbox_colors)

if __name__ == '__main__':
    try:
        print("Cone Detection: running @", os.getcwd())
        rospy.init_node('color_det', anonymous=False)
        rospy.Subscriber('Frame_BBox', frame_msg, callback)
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
