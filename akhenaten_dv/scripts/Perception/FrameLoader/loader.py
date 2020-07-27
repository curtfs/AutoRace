#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
from datasets import LoadImages
import time
import sys
import numpy as np
load_path = "./src/akhenaten_dv/scripts/Perception/FrameLoader/input"
bridge = CvBridge()
#ms = sys.argv[1]
def frame_loader():
    VideoRaw = rospy.Publisher('input_frames', Image)
    dataset = LoadImages(load_path, img_size=512)
    while not rospy.is_shutdown():
        for path, img, img0, cap in dataset:
            msg_frame = bridge.cv2_to_imgmsg(img0)
            VideoRaw.publish(msg_frame)
            print("Published: ", path)
            time.sleep(float(ms))

def frame_callback(ros_data):
    np_arr = np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    VideoRaw = rospy.Publisher('input_frames', Image)
    msg_frame = bridge.cv2_to_imgmsg(image_np)
    VideoRaw.publish(msg_frame)


if __name__ == '__main__':
    try:
        rospy.init_node('frame_loader', anonymous=False)
        #frame_loader()
        #rospy.Subscriber('/left/image_rect_color/compressed', CompressedImage, frame_callback)
        rospy.Subscriber('/left/image_rect_color/compressed', CompressedImage, frame_callback)
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
