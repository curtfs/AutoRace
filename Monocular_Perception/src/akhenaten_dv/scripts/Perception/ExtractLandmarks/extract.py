#!/usr/bin/env python
import rospy
import tensorflow as tflow
from sensor_msgs.msg import Image
import cv2
from akhenaten_dv.msg import frame_msg
from akhenaten_dv.msg import landmarks
import math
import numpy as np
import time
import os
from cv_bridge import CvBridge, CvBridgeError
import datetime
from geometry_msgs.msg import PointStamped
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

x = datetime.datetime.now()
cones_colors = [(255,0,0),(0,255,255),(0,165,255), (200,200,200)]
kp_colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (127, 255, 127), (255, 127, 127)]
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
# fontScale 
fontScale = 0.5
# Blue color in BGR 
color = (0, 0, 0) 
# Line thickness of 2 px 
thickness = 1
output_path = 'src/akhenaten_dv/saved_frames/'
run_name = str(x).split(' ')[0]
if not os.path.exists(output_path+run_name):
    os.mkdir(output_path+run_name)
class PnpAlgorithm():
    def __init__(self):
        # 3D model points.
        self.model_points = np.array([
                                    (0      , 324.358, 0.0),
                                    (-30.988 , 274.32, 0.0),
                                    (30.988 , 274.32, 0.0),
                                    (-48.768 , 172.974, 0.0),
                                    (48.768 , 172.974, 0.0),
                                    (-75.184 , 13.876, 0.0),
                                    (75.184 , 13.876, 0.0)
                                ])

        # Camera internals
        #ZED Camera
        self.camera_matrix = np.array(
                                [[343.6113586425781, 0,338.4967041015625,],
                                [0, 343.6113586425781, 193.01895141601562],
                                [0, 0, 1]], dtype = "double"
                                )
        #AMZ Camera
        # self.camera_matrix = np.array(
        #                         [[535.4, 0,512],
        #                         [0, 539.2, 360],
        #                         [0, 0, 1]], dtype = "double"
        #                         )
        self.dist_coeffs = np.zeros((1,4))
        #self.dist_coeffs = np.array([0.262383, -0.953104, -0.005358, 0.002628, 1.163314], dtype = "double")


    def run(self, im_points):
        #2D image points.
        image_points = []
        #print("Cone points: ")
        for pt in im_points:
            image_points.append((pt[0],pt[1]))
        image_points = np.array(image_points)
        (success, rotation_vector, translation_vector, inliners) = cv2.solvePnPRansac(self.model_points, image_points, self.camera_matrix, self.dist_coeffs, reprojectionError=80, iterationsCount = 100000000, confidence=0.9, flags = 0)

        translation_vector = translation_vector.reshape(1,3)
        #translation_vector[0][1] = translation_vector[0][2]
        translation_vector[0][1] = 0
        translation_vector=translation_vector/1000
        return translation_vector[0], rotation_vector, success


# Publish cones as numpy array with the cone types and positions
def publish_cones(local_map, transform_to_map=False):
    
    if local_map.shape[0] == 0:
        
        return
    
    cone_publisher = rospy.Publisher('/perception_cones', numpy_msg(Floats), queue_size=10)
    # global listener
    # listener.lookupTransform('zed_left_camera','map', rospy.Time(0))
    
    transformed_cones = np.array([])

    for cone in local_map:
        
        conePoint_local = PointStamped()
        conePoint_local.header.frame_id = 'zed_left_camera'
        # conePoint_local.point.x = cone[2]
        # conePoint_local.point.y = -cone[0]
        # conePoint_local.point.z = -cone[1]
        conePoint_local.point.x = cone[2]
        conePoint_local.point.y = -cone[0]
        conePoint_local.point.z = -cone[1]
        conePoint_global = PointStamped()
        # conePoint_global = listener.transformPoint("/map", conePoint_local)

        # global_cone = np.array([conePoint_global.point.x, conePoint_global.point.y, conePoint_global.point.z, float(cone[3])], dtype=np.float32).reshape(1,4)
        local_cone = np.array([conePoint_local.point.x, conePoint_local.point.y, conePoint_local.point.z, float(cone[3])], dtype=np.float32).reshape(1,4)

        if transform_to_map:
            # if transformed_cones.shape[0] > 0:
            #     transformed_cones = np.vstack((transformed_cones, global_cone))
            # else:
            #     transformed_cones = global_cone
            print()
        else:
            if transformed_cones.shape[0] > 0:
                transformed_cones = np.vstack((transformed_cones, local_cone))
            else:
                transformed_cones = local_cone       
    cone_publisher.publish(transformed_cones.flatten())



def setup_bboxes(bboxes_corrupt):
    BBoxes=[]
    for i in range(0, len(bboxes_corrupt), 4):
        BBoxes.append([bboxes_corrupt[i],bboxes_corrupt[i+1],bboxes_corrupt[i+2],bboxes_corrupt[i+3]])
    return BBoxes


bridge = CvBridge()
r=0
fframes_publisher = rospy.Publisher('final_frames', Image)
def export_frames(data, tvec, success):
    global r
    img0 = bridge.imgmsg_to_cv2(data.image_frame)
    bboxes = setup_bboxes(data.BBox)
    kps = data.Kps
    colors = data.Color
    c = 0
    for i in range(0, int(len(tvec)),3):
        color = colors[c]
        [x,y,w,h] = bboxes[c]
        c+=1
        img0 = cv2.putText(img0, 'x: '+str(round(tvec[i],3))+'. y: '+str(round(tvec[i+2],3)), (x,y), font, fontScale, color, thickness, cv2.LINE_AA)  
        if True:
            cv2.rectangle(img0, (x,y), (x+w,y+h), cones_colors[color], 1)
        else:
            cv2.rectangle(img0, (x,y), (x+w,y+h), cones_colors[3], 1)
        if success[int(i/3)]:
            if True:
                cv2.circle(img0, (int(img0.shape[1]/2 + tvec[i]/60*img0.shape[1]), int(img0.shape[0]-tvec[i+2]/40*img0.shape[0])), 4, cones_colors[color], -1)
            else:
                cv2.circle(img0, (int(img0.shape[1]/2 + tvec[i]/60*img0.shape[1]), int(img0.shape[0]-tvec[i+2]/40*img0.shape[0])), 4, cones_colors[3], -1)
    bboxes = data.BBox
    k=0
    p=0
    for i in range(0, int(len(kps)),14):
        x = bboxes[p]
        y = bboxes[p+1]
        w = bboxes[p+2]
        h = bboxes[p+3]
        p+=4
        for j in range(0,14,2):
            pt = [kps[i+j]*w+x, kps[i+j+1]*h+y]
            cv2.circle(img0, (int(pt[0]), int(pt[1])), 2, (255,0,0), -1)
            k=+1

    cv2.imwrite(output_path+run_name+"/"+str(r)+'.jpg', img0)
    msg_frame = bridge.cv2_to_imgmsg(img0)
    fframes_publisher.publish(msg_frame)
    r+=1

pnp = PnpAlgorithm()
landmarks_publisher = rospy.Publisher('localized_landmarks', landmarks)
landmarks_message = landmarks()
def callback(data):
    landmarks_message.success = []
    print("Landmark Extraction: Recieved frame")
    st = time.time()
    points = data.Kps
    bboxes = setup_bboxes(data.BBox)
    colors = data.Color
    landmarks_tvec=[]
    landmarks_rvec=[]
    k=0
    tvecs_x = []
    tvecs_y = []
    tvecs_z = []
    color_map = []
    for i in range(0, int(len(points)),14):
        kpoints = []
        for j in range(0,14,2):
            [x,y,w,h] = bboxes[k]
            xy = [points[i+j]*w+x, points[i+j+1]*h+y]
            kpoints.append(xy)
        color = colors[k]
        k+=1
        tvec, rvec, success = pnp.run(kpoints)
        if success:
            tvecs_x.append(tvec[0])
            tvecs_y.append(tvec[1])
            tvecs_z.append(tvec[2])
            color_map.append(color)
            landmarks_tvec.extend(tvec)
            landmarks_rvec.extend(rvec[0])
            landmarks_rvec.extend(rvec[2])
            landmarks_rvec.extend(rvec[1])
            landmarks_message.success.append(success)
    end1 = time.time()
    landmarks_message.tvec = landmarks_tvec
    landmarks_message.rvec = landmarks_rvec
    print("Landmark extracted in ", (end1 - st), ' seconds')
    landmarks_publisher.publish(landmarks_message)
    cones_map = np.array((tvecs_x,tvecs_y,tvecs_z,color_map)).T
    publish_cones(cones_map)
    export_frames(data, landmarks_tvec, landmarks_message.success)


if __name__ == '__main__':
    try:
        print("Extract Landmark: running @", os.getcwd())
        rospy.init_node('Extract_Landmarks', anonymous=False)
        rospy.Subscriber('Frame_BBox_Color_Kps', frame_msg, callback)
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
