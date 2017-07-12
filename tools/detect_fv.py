# -*- coding: utf-8 -*-
from __future__ import division
import init_path
import os.path as osp
import selective_search as ss
from config import cfg
import cv2
import glob
import nms
import extract_features as extfeat
from matplotlib import pyplot as plt
from sklearn.externals import joblib

def load_config():
    # path params
    global model_dir, output_dir, test_dir
    model_dir = cfg.METHOD.FV.MODEL_DIR
    output_dir = cfg.OUTPUT_DIR
    test_dir = cfg.DATASET.TEST_DIR 
    # image detect size
    global im_detect_maxlen, im_size
    im_detect_maxlen = cfg.DETECT.IMAGE_MAX_LENGTH
    im_size = cfg.TRAIN.SAMPLE_SIZE[cfg.TRAIN.SCHEME]
    # selective param
    global color_spaces, ks, feature_masks
    color_spaces = cfg.SS.COLOR_SPACES
    ks = cfg.SS.KS
    feature_masks = cfg.SS.FEATURE_MARKS
    
def image_rescale(image_path):
    # rescale to suitable scale
    img = cv2.imread(image_path)
    imshp = img.shape
    maxlen = max(imshp[0],imshp[1])
    scale = 1.0
    if maxlen > im_detect_maxlen:
        scale = im_detect_maxlen / maxlen
    detectimg = cv2.resize(img,(int(imshp[1]*scale),int(imshp[0]*scale)))
    return detectimg

def detect(image_path):
    image = image_rescale(image_path)
    # transform bgr to rgb
    im = image[:,:,(2,1,0)]
    regions = ss.selective_search(im, color_spaces, ks, feature_masks)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, aspect='equal')
    locations = []
    for v, (i0, j0, i1, j1) in regions:
        w = j1 - j0
        h = i1 - i0
        # check
        if w < 20 or h < 20 or w > 50 or h > 50 or w/h > 1.2 or h/w > 1.2:
            continue
        # extract pathch
        patch = image[i0:i1,j0:j1,:]
        patch = cv2.resize(patch,im_size)
        # extract fv feature
        _, des = extfeat.extract_sift_from_image(patch)
        if des is None or len(des) == 0:
            continue
        X_fv = extfeat.fv_feature([des],retrain=False,suffix='FV')
        # load classifier
        clf = joblib.load(osp.join(model_dir,'fv_rf.pkl'))
        pred = clf.predict_proba(X_fv)
        if pred[0][1] > 0.50:
            locations.append((j0, i0, pred[0][1], w, h))
    # nms
    locations = nms.nms(locations)
    # plot
    for (x, y, score, w, h) in locations:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    img_name = image_path.split('/')[-1]
    cv2.imwrite(osp.join(output_dir,'fv_'+img_name), image)
    #cv2.imshow('detected result',image)
    #cv2.waitKey()

def run():
    load_config()
    # detect
    for image_name in glob.iglob(osp.join(test_dir,'*.jpg')):
        print '[Detect FV] Process: ',image_name
        detect(image_name)
        
if __name__ == '__main__':
    run()
    