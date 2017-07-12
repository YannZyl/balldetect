# -*- coding: utf-8 -*-
from __future__ import division
import init_path
import glob
import cv2
import os
import random
import spm
import numpy as np
import extract_features as extfeats
from config import cfg
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

def load_config():
    # path params
    global model_dir, feat_dir, pos_dir, neg_dir
    model_dir = cfg.METHOD.SPM.MODEL_DIR
    feat_dir = cfg.METHOD.SPM.FEATURE_DIR
    pos_dir = cfg.DATASET.POSITIVE_DIR
    neg_dir = cfg.DATASET.NEGATIVE_DIR
    # image params
    global im_size
    im_size = cfg.TRAIN.SAMPLE_SIZE[cfg.TRAIN.SCHEME]
    # train params
    global spm_level, spm_vocs
    spm_level = cfg.METHOD.SPM.LEVEL
    spm_vocs = cfg.METHOD.SPM.CLUSTERS
    print '[Train SPM] Load global configure done.'
    
def clean_dir(clean=False):
    # clean model dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    else:
        for files in glob.iglob(model_dir+"/*"):
            os.remove(files)
    # clean feature dir
    if not os.path.exists(feat_dir):
        os.mkdir(feat_dir)
    else:
        for files in glob.iglob(feat_dir+"/*"):
            os.remove(files)
    print '[Train SPM] Clean model directory done.'
            
def shuffle_data(X, Y):
    index = range(len(X))
    random.shuffle(index)
    X = [X[i] for i in index]
    Y = [Y[i] for i in index]
    print '[Train SPM] Shuffle data done.'
    return X, Y  
    
def load_data():
    X, Y = [], []
    for posimg in glob.iglob(pos_dir+'/*'):
        im = cv2.imread(posimg)
        im = cv2.resize(im, im_size)
        X.append(im)
        Y.append(1)
    for negimg in glob.iglob(neg_dir+'/*'):
        im = cv2.imread(negimg)
        im = cv2.resize(im, im_size)
        X.append(im)
        Y.append(-1)
    print '[Train SPM] Load data done.'
    return X, Y  
    
# opencv svm
def train_svm(X, Y):
    # generate train and test set
    data_nums = X.shape[0]
    train_nums = int(data_nums * 0.7)
    X_train = X[0:train_nums, :]
    Y_train = Y[0:train_nums, :]
    X_test = X[train_nums: , :]
    Y_test = Y[train_nums: , :]
    # configure svm
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    # train linear svm
    svm.train(X_train, cv2.ml.ROW_SAMPLE, Y_train)
    svm.save(os.path.join(model_dir,'svm_SPM.xml'))
    # test
    _, pred = svm.predict(X_test)
    right = 0
    for idx, result in enumerate(pred):
        if int(result[0]) == int(Y_test[idx]):
            right += 1
    print '[Train SPM] Linear SPM SVM acc:', right / Y_test.shape[0]
    print '[Train SPM] Linear SPM SVM train done.'

# sklearn random forests
def train_rf(X, Y):
   # generate train and test set
    data_nums = X.shape[0]
    train_nums = int(data_nums * 0.7)
    X_train = X[0:train_nums, :]
    Y_train = Y[0:train_nums]
    X_test = X[train_nums: , :]
    Y_test = Y[train_nums:]
    # configure svm
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X_train, Y_train)
    joblib.dump(clf, os.path.join(model_dir,'spm_rf.pkl')) 
    # test
    sc = clf.score(X_test,Y_test)
    print '[Train SPM] Random SPM Forests acc:', sc
    print '[Train SPM] Random SPM Forests train done.'
    
def run():
    # load global configue
    load_config()
    # clean model directory
    clean_dir(clean=True)
    # load data
    data, labels = load_data()
    # shuffle data
    data, labels = shuffle_data(data, labels)
    # extract sift
    sift_des, sift_pts, sift_ind = extfeats.extract_sift_from_imlist(data)
    # extract spm feature
    image_info = {'des':sift_des, 'img':data, 'kps':sift_pts}
    spm_feats = spm.spm_fit(image_info, level=spm_level, nclusters=spm_vocs) 
    spm_labels = np.array([labels[idx] for idx in sift_ind], dtype='int')
    joblib.dump([spm_feats,spm_labels],os.path.join(feat_dir,'spm_feat.pkl'))
    #feats,labels = joblib.load(os.path.join(feat_dir,'feat.pkl'))
    # train classify
    train_rf(spm_feats, spm_labels)

if __name__ == '__main__':
    run()    
    