# -*- coding: utf-8 -*-
from __future__ import division
import init_path
import cv2
import numpy as np
import nms
import glob
import os.path as osp 
import extract_features as extfeat
from config import cfg
from sklearn.externals import joblib

def load_config():
    # path params
    global model_dir, output_dir, test_dir
    model_dir = cfg.METHOD.OURS.MODEL_DIR
    output_dir = cfg.OUTPUT_DIR
    test_dir = cfg.DATASET.TEST_DIR 
    # image detect size
    global im_detect_maxlen
    im_detect_maxlen = cfg.DETECT.IMAGE_MAX_LENGTH
    # detect param
    global bow_ncmp, win_size
    bow_ncmp = cfg.METHOD.OURS.BOW_NCOMPONENTS
    win_size = cfg.DETECT.RESIZE_SIZE[cfg.DETECT.SCHEME]
    
def init_detector(xml_path):
    # load svm through xml file  
    clf = cv2.ml.SVM_load(xml_path)   
    rho, alpha, _ = clf.getDecisionFunction(0)
    sv = clf.getSupportVectors()
    mdetect = -1 * alpha.dot(sv)[0]
    mdetect = np.concatenate((mdetect,[rho]),axis=0)
    # initial hog descriptor
    hog = extfeat.get_hog_descriptor()       
    hog.setSVMDetector(mdetect)
    return hog
    
def image_rescale(test_image):
    # rescale to suitable scale
    img = cv2.imread(test_image)
    maxlen = max(img.shape[0],img.shape[1])
    scale = 1.0
    if maxlen > im_detect_maxlen:
        scale = im_detect_maxlen / maxlen
    detectimg = cv2.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)))
    detectimg = detectimg.astype('uint8')
    return img   
    
def detect(imgpath):
    # load config
    load_config()
    # Hog multiscale detection
    image = image_rescale(imgpath)
    # initalize hog detector
    hog = init_detector(osp.join(model_dir,'svm_HoG.xml'))
    # multiscale detect
    locations, weights = hog.detectMultiScale(image, winStride=(8,8),padding=(16,16), scale=1.05) 
    # modify bounding box
    new_locations = []
    for i in xrange(len(locations)):
        x = int((locations[i,0] + locations[i,2] * 0.1))
        y = int((locations[i,1] + locations[i,3] * 0.07))
        w = int((locations[i,2] * 0.8) )
        h = int((locations[i,3] * 0.8) )
        # setect bbox
        if x >= 0 and y >= 0 and x+w < image.shape[1] and y+h < image.shape[0] and w/h<1.5 and h/w<1.5:
            new_locations.append([x,y,weights[i,0],w,h])              
    # nms and plot
    new_locations = nms.nms(new_locations)
    for (x, y, score, w, h) in new_locations:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)  
    img_name = imgpath.split('/')[-1]
    cv2.imwrite(osp.join(output_dir,'hog_'+img_name), image)
    #cv2.imshow('detected result',image)
    #cv2.waitKey()

def run():
    # load config
    load_config()
    # detect
    for image_name in glob.iglob(osp.join(test_dir,'*.jpg')):
        print '[Detect HOG] Process: ',image_name
        detect(image_name)
        
if __name__ == '__main__':
    run()