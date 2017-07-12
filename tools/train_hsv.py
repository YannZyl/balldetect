# -*- coding: utf-8 -*-
from __future__ import division
import init_path
import glob
import cv2
import os
import random
import numpy as np
from config import cfg
import extract_features as extfeat
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

def load_config():
    # path params
    global model_dir, feat_dir, pos_dir, neg_dir
    model_dir = cfg.METHOD.HSV.MODEL_DIR
    feat_dir = cfg.METHOD.HSV.FEATURE_DIR
    pos_dir = cfg.DATASET.POSITIVE_DIR
    neg_dir = cfg.DATASET.NEGATIVE_DIR
    # image params
    global im_size
    im_size = cfg.TRAIN.SAMPLE_SIZE[cfg.TRAIN.SCHEME]
    print '[Train HSV] Load global configure done.'

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
    print '[Train HSV] Clean model directory done.'

def load_data():
    X, Y = [], []
    for posimg in glob.iglob(pos_dir+'/*'):
        im = cv2.imread(posimg)
        im = cv2.resize(im, im_size)
        hsv_feat = extfeat.colorHist(im)
        X.append(hsv_feat)
        Y.append(1)
    for negimg in glob.iglob(neg_dir+'/*'):
        im = cv2.imread(negimg)
        im = cv2.resize(im, im_size)
        hsv_feat = extfeat.colorHist(im)
        X.append(hsv_feat)
        Y.append(-1)
    print '[Train HSV] Load data and convert to hsv histogram done.'
    return X, Y
    
def shuffle_data(X, Y):
    index = range(len(X))
    random.shuffle(index)
    X = [X[i] for i in index]
    Y = [Y[i] for i in index]
    print '[Train HSV] Shuffle data done.'
    return X, Y   

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
    svm.save(os.path.join(model_dir,'svm_HSV.xml'))
    # test
    _, pred = svm.predict(X_test)
    right = 0
    for idx, result in enumerate(pred):
        if int(result[0]) == int(Y_test[idx]):
            right += 1
    print '[Train HSV] Linear HSV SVM acc:', right / Y_test.shape[0]
    print '[Train HSV] Linear HSV SVM train done.'

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
    joblib.dump(clf, os.path.join(model_dir,'hsv_rf.pkl')) 
    # test
    sc = clf.score(X_test,Y_test)
    print '[Train HSV] Random HSV Forests acc:', sc
    print '[Train HSV] Random HSV Forests train done.'
    
def run():
    # load global params 
    load_config()
    # clean directories
    clean_dir(clean=True)
    # load data and convert to hsv histogram
    hsv_feats, labels = load_data()
    # shuffle data and save data
    hsv_feats, labels = shuffle_data(hsv_feats, labels)
    hsv_feats = np.array(hsv_feats,dtype='float32')
    labels = np.array(labels, dtype='int')
    joblib.dump([hsv_feats, labels],os.path.join(feat_dir,'hsv_feats.pkl'))
    # train linear svm classify
    train_rf(hsv_feats, labels)

if __name__ == '__main__':
    run()