#!/usr/bin/python3
import torch
import numpy as np
import sys
from utils_k import vis_tensor_and_save, prep_image,vis_tensor_and_save_2
from keypoint_net import KeypointNet
import argparse
import os
from os.path import isfile, join
import random
import tempfile
import time
import copy
import multiprocessing
import subprocess
import shutil
import cv2
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from PIL import Image, ImageDraw, ImageFont

import torchvision
from models import Darknet
from utils.datasets import ImageLabelDataset
from utils.nms import nms
from utils.utils import xywh2xyxy, calculate_padding

import warnings
from tqdm import tqdm
from color_classification_image import predict_color

warnings.filterwarnings("ignore")

detection_tmp_path = "/tmp/detect/"

visualizer = plt.figure()
# ax = visualizer.add_subplot(111)
# ax.set_xlim(-10, 10)
# ax.set_ylim(0, 20)

plt.ion()
def main(target_path,output_path,weights_path,model_cfg,kpoints_model_path,conf_thres,nms_thres,xy_loss,wh_loss,no_object_loss,object_loss,vanilla_anchor):

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    random.seed(0)
    torch.manual_seed(0)
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    model = Darknet(config_path=model_cfg,xy_loss=xy_loss,wh_loss=wh_loss,no_object_loss=no_object_loss,object_loss=object_loss,vanilla_anchor=vanilla_anchor)

    # Load weights
    model.load_weights(weights_path, model.get_start_weight_dim())
    model.to(device, non_blocking=True)
    print(device)
    #Keypoints part
    #****************************##******************##*********************##***************##*****************##****************
    output_path_k = "outputs_k/visualization/"
    flip=False
    rotate=False
    #model_path = "/home/abdouseif/fsai-cv/MIT-Driverless-CV-TrainingInfra-master/CVC-YOLOv3/outputs_k/april-2020-experiments/testTwo/55_loss_0.01.pt"
    model_path=kpoints_model_path
    model_filepath = model_path
    model_k = KeypointNet()
    model_k.load_state_dict(torch.load(model_filepath).get('model'))
    model_k.eval()

    #****************************##******************##*********************##***************##*****************##****************
    startG = timer()
    print("Detection Started")
    detect(target_path,
           output_path,
           model,
           device=device,
           conf_thres=conf_thres,
           nms_thres=nms_thres,
           model_k=model_k)
    endG = timer()
    print("Landmark Detection finished: ", (endG-startG), " seconds")

def single_img_detect(target_path,output_path,mode,model,device,conf_thres,nms_thres,model_k1):
    start = timer()
    print("Retrieving Image")
    rnd_img = Image.open(target_path).convert('RGB')
    img = Image.open(target_path).convert('RGB')
    w, h = img.size
    new_width, new_height = model.img_size()
    pad_h, pad_w, ratio = calculate_padding(h, w, new_height, new_width)
    img = torchvision.transforms.functional.pad(img, padding=(pad_w, pad_h, pad_w, pad_h), fill=(127, 127, 127), padding_mode="constant")
    img = torchvision.transforms.functional.resize(img, (new_height, new_width))

    bw = model.get_bw()
    if bw:
        img = torchvision.transforms.functional.to_grayscale(img, num_output_channels=1)

    img = torchvision.transforms.functional.to_tensor(img)
    img = img.unsqueeze(0)
    end = timer()
    print("Retrieving finished: ", (end-start), " seconds")

    with torch.no_grad():
        start = timer()
        print("YOLO Detection Started")
        #model.eval()
        img = img.to(device, non_blocking=True)
        # output,first_layer,second_layer,third_layer = model(img)
        output = model(img)
        for detections in output:
            print(detections[:,0])
            detections = detections[detections[:, 4] > conf_thres]
            
            box_corner = torch.zeros((detections.shape[0], 4), device=detections.device)

            xy = detections[:, 0:2]
            wh = detections[:, 2:4] / 2

            box_corner[:, 0:2] = xy - wh
            box_corner[:, 2:4] = xy + wh
            probabilities = detections[:, 4]
            nms_indices = nms(box_corner, probabilities, nms_thres)
            main_box_corner = box_corner[nms_indices]
            if nms_indices.shape[0] == 0:
                continue
        img_with_boxes = Image.open(target_path)
        image2 = cv2.imread(target_path)
        end = timer()
        print("YOLO Detection finished: ", (end-start), " seconds")

        start = timer()
        print("Keypoint Regression + PnP Started")
        ax = visualizer.add_subplot(111)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-1, 15)
        tveclist = []
        successlist = []
        for i in range(len(main_box_corner)):
            x0 = main_box_corner[i, 0].to('cpu').item() / ratio - pad_w
            y0 = main_box_corner[i, 1].to('cpu').item() / ratio - pad_h
            x1 = main_box_corner[i, 2].to('cpu').item() / ratio - pad_w
            y1 = main_box_corner[i, 3].to('cpu').item() / ratio - pad_h
            im_crop = img_with_boxes.crop((x0, y0, x1, y1))
            #im_crop.save(target_path+str(i)+".jpg")
            image_path = target_path+str(i)+".jpg"

            image_filepath = image_path
            img_name = '_'.join(image_filepath.split('/')[-1].split('.')[0].split('_')[-5:])
            img_size=80
            image_size = (img_size, img_size)
            image = cv2.cvtColor(np.array(im_crop),cv2.COLOR_RGB2BGR)
            image2 = cv2.cvtColor(np.array(im_crop),cv2.COLOR_RGB2BGR)
            h, w, _ = image.shape
            image = prep_image(image=image,target_image_size=image_size)
            image = (image.transpose((2, 0, 1)) / 255.0)[np.newaxis, :]
            image = torch.from_numpy(image).type('torch.FloatTensor')
            output = model_k1(image)
            color=predict_color(image2)
            print(color)
            out = np.empty(shape=(0, output[0][0].shape[2]))
            for o in output[0][0]:
                chan = np.array(o.cpu().data)
                cmin = chan.min()
                cmax = chan.max()
                chan -= cmin
                chan /= cmax - cmin
                out = np.concatenate((out, chan), axis=0)
            output_path_k = "outputs_k/visualization/"
            #cv2.imwrite(output_path_k + img_name + "_hm.jpg", out * 255)
            #print(f'please check the output image here: {output_path_k + img_name + "_hm.jpg", out * 255}')
            image = cv2.cvtColor(np.array(im_crop),cv2.COLOR_RGB2BGR)
            h, w, _ = image.shape
            #vis_tensor_and_save(image=image, h=h, w=w, tensor_output=output[1][0].cpu().data, image_name=img_name, output_uri=output_path_k)
            h2, w2, _ = image2.shape
            vis_tensor_and_save_2(image=image2,x_translation=x0,y_translation=y0, h=h, w=w, tensor_output=output[1][0].cpu().data, image_name=img_name, output_uri=output_path_k,target_path=output_path+"fs.jpeg")
            (tvec, rvec, success) = estimatePose(image=image2,x_translation=x0,y_translation=y0, h=h, w=w, tensor_output=output[1][0].cpu().data)
            tveclist.append(tvec)
            successlist.append(success)

            #if tvec[0] < 50000 and tvec[1] < 50000 and success:
            #    ax.scatter(tvec[0], tvec[1])
                
            image2 = cv2.imread(output_path+"fs.jpeg")
        end = timer()
        print("Keypoint Regression + PnP finished: ", (end-start), " seconds")
        #plt.pause(0.05)
        #plt.show()
        #plt.waitforbuttonpress()
        #plt.clf()

        
        # get a font
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 15)


        draw = ImageDraw.Draw(img_with_boxes)
        w, h = img_with_boxes.size

        #draw.rectangle((700, 500, 1024, 672), fill = "black", outline="black")

        for i in range(len(main_box_corner)):
            x0 = main_box_corner[i, 0].to('cpu').item() / ratio - pad_w
            y0 = main_box_corner[i, 1].to('cpu').item() / ratio - pad_h
            x1 = main_box_corner[i, 2].to('cpu').item() / ratio - pad_w
            y1 = main_box_corner[i, 3].to('cpu').item() / ratio - pad_h
            if successlist[i] and (tveclist[i])[0] < 6 and (tveclist[i])[1] < 20:
                # draw text, full opacity
                draw.ellipse([((tveclist[i][0]+6)/12*324+380,(-tveclist[i][1])/8*172+690),((tveclist[i][0]+6)/12*324+385,(-tveclist[i][1])/8*172+695)], fill="red")
                draw.text((x0-40,y0-20), "x= "+str(tveclist[i][0])+", y= "+str(tveclist[i][1]), font=fnt, fill=(0,0,0,255))
            draw.rectangle((x0, y0, x1, y1), outline="black")


        if mode == 'image':
            img_with_boxes.save(os.path.join(output_path,target_path.split('/')[-1]))
            return os.path.join(output_path,target_path.split('/')[-1])
        else:
            img_with_boxes.save(target_path)
            return target_path



def estimatePose(image, x_translation, y_translation, h, w, tensor_output):
    #2D image points.
    image_points = []
    #print("Cone points: ")
    for pt in np.array(tensor_output):
        #print(pt[0]*w,pt[1]*h)
        image_points.append((pt[0]*w+x_translation,pt[1]*h+y_translation))

    image_points = np.array(image_points)
    # 3D model points.

    model_points = np.array([
                                (0      , 324.358, 0.0),
                                (30.988 , 274.32, 0.0),
                                (-30.988 , 274.32, 0.0),
                                (48.768 , 172.974, 0.0),
                                (-48.768 , 172.974, 0.0),
                                (75.184 , 13.876, 0.0),
                                (-75.184 , 13.876, 0.0)
                            ])

    # Camera internals
    camera_matrix = np.array(
                            [[535.4, 0,512],
                            [0, 539.2, 360],
                            [0, 0, 1]], dtype = "double"
                            )


    #dist_coeffs = np.zeros((1,4))
    dist_coeffs = np.array([0.262383, -0.953104, -0.005358, 0.002628, 1.163314], dtype = "double")
    (success, rotation_vector, translation_vector, inliners) = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs, reprojectionError=100, iterationsCount = 10, confidence=0.9, flags = 0)


    #print ("Rotation Vector:\n {0}".format(rotation_vector))
    #print ("Translation Vector:\n {0}".format(translation_vector))

    #print ("Inliners:\n {0}".format(success))

    translation_vector = translation_vector.reshape(1,3)
    translation_vector[0][1] = translation_vector[0][2]
    translation_vector[0][2] = 0
    translation_vector=translation_vector/1000


    return translation_vector[0], rotation_vector, success

def detect(target_path,
           output_path,
           model,
           device,
           conf_thres,
           nms_thres,
           model_k):

        target_filepath = target_path
        img_formats = ['.jpg', '.jpeg', '.png', '.tif']
        vid_formats = ['.mov', '.avi', '.mp4']

        mode = None

        if os.path.splitext(target_filepath)[-1].lower() in img_formats:
            mode = 'image'

        elif os.path.splitext(target_filepath)[-1].lower() in vid_formats:
            mode = 'video'

        print("Detection Mode is: " + mode)

        raw_file_name = target_filepath.split('/')[-1].split('.')[0].split('_')[-4:]
        raw_file_name = '_'.join(raw_file_name)

        if mode == 'image':
            start = timer()
            detection_path = single_img_detect(target_path=target_filepath,output_path=output_path,mode=mode,model=model,device=device,conf_thres=conf_thres,nms_thres=nms_thres,model_k1=model_k)
            end = timer()
            print("single_img_detect executed", end-start)
            #print(f'Please check output image at {detection_path}')

        elif mode == 'video':
            if os.path.exists(detection_tmp_path):
                shutil.rmtree(detection_tmp_path)  # delete output folder
            os.makedirs(detection_tmp_path)  # make new output folder

            vidcap = cv2.VideoCapture(target_filepath)
            success,image = vidcap.read()
            count = 0

            while success:
                cv2.imwrite(detection_tmp_path + "/frame%d.jpg" % count, image)     # save frame as JPEG file
                success,image = vidcap.read()
                count += 1

            #Find OpenCV version
            (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

            if int(major_ver)  < 3 :
                fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
                print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
            else :
                fps = vidcap.get(cv2.CAP_PROP_FPS)
                print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
            vidcap.release();

            frame_array = []
            files = [f for f in os.listdir(detection_tmp_path) if isfile(join(detection_tmp_path, f))]

            #for sorting the file names properly
            files.sort(key = lambda x: int(x[5:-4]))
            for i in tqdm(files,desc='Doing Single Image Detection'):
                filename=detection_tmp_path + i

                detection_path = single_img_detect(target_path=filename,output_path=output_path,mode=mode,model=model,device=device,conf_thres=conf_thres,nms_thres=nms_thres,model_k1=model_k)
                #reading each files
                img = cv2.imread(detection_path)
                height, width, layers = img.shape
                size = (width,height)
                frame_array.append(img)

            local_output_uri = output_path + raw_file_name + ".mp4"

            video_output = cv2.VideoWriter(local_output_uri,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

            for frame in tqdm(frame_array,desc='Creating Video'):
                # writing to a image array
                video_output.write(frame)
            video_output.release()
            print(f'please check output video at {local_output_uri}')
            shutil.rmtree(detection_tmp_path)
        #print("Please go to the link below to check the detection output file: ")
        #print(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})
    parser.add_argument('--model_cfg', type=str, default='model_cfg/yolo_baseline.cfg')
    parser.add_argument('--kpoints_model_path', type=str,default='model_cfg/55_loss_0.01.pt')
    parser.add_argument('--target_path', type=str, help='path to target image/video')
    parser.add_argument('--output_path', type=str, default="outputs/visualization/")
    parser.add_argument('--weights_path', type=str, help='path to weights file')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.25, help='IoU threshold for non-maximum suppression')

    add_bool_arg('vanilla_anchor', default=False, help="whether to use vanilla anchor boxes for training")
    ##### Loss Constants #####
    parser.add_argument('--xy_loss', type=float, default=2, help='confidence loss for x and y')
    parser.add_argument('--wh_loss', type=float, default=1.6, help='confidence loss for width and height')
    parser.add_argument('--no_object_loss', type=float, default=25, help='confidence loss for background')
    parser.add_argument('--object_loss', type=float, default=0.1, help='confidence loss for foreground')

    opt = parser.parse_args()

    main(target_path=opt.target_path,
         output_path=opt.output_path,
         weights_path=opt.weights_path,
         model_cfg=opt.model_cfg,
         kpoints_model_path=opt.kpoints_model_path,
         conf_thres=opt.conf_thres,
         nms_thres=opt.nms_thres,
         xy_loss=opt.xy_loss,
         wh_loss=opt.wh_loss,
         no_object_loss=opt.no_object_loss,
         object_loss=opt.object_loss,
         vanilla_anchor=opt.vanilla_anchor)
