#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 8th July 2018 - before Google inside look 2018 :)
# -------------------------------------------------------------------------

import cv2
import color_histogram_feature_extraction
import knn_classifier
import os
import os.path
import sys
from timeit import default_timer as timer

# read the test image
# try:
#     source_image = cv2.imread(sys.argv[1])
# except:
#     source_image = cv2.imread('vid_2_frame_4303_0.jpg')
# prediction = 'n.a.'
def predict_color(img):
    #checking whether the training data is ready
    
    # PATH = './training.data'

    # if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    #     print ('training data is ready, classifier is loading...')
    # else:
    #     print ('training data is being created...')
    #     open('training.data', 'w')
    #     color_histogram_feature_extraction.training()
    #     print ('training data is ready, classifier is loading...')

    # get the prediction
    #start = timer()
    color_histogram_feature_extraction.color_histogram_of_test_image(img)
    prediction = knn_classifier.main('training.data', 'test.data')
    return prediction
    #end = timer()
    #print('Detected color is:', prediction,' in ', end-start,' second')
    # cv2.putText(
    #     source_image,
    #     'Prediction: ' + prediction,
    #     (15, 45),
    #     cv2.FONT_HERSHEY_PLAIN,
    #     3,
    #     200,
    #     )

    # Display the resulting frame
    # cv2.imshow('color classifier', source_image)
    # cv2.waitKey(0)		
