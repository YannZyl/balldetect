# -*- coding: utf-8 -*-
from __future__ import division
import init_path
import glob
import cv2
import os
import numpy as np
import random
import visualize as vis
import extract_features as extfeat
from config import cfg
from sklearn.externals import joblib
from sklearn import manifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier

def load_config():
    # path params
    global model_dir, feat_dir, pos_dir, neg_dir
    model_dir = cfg.METHOD.OURS.MODEL_DIR
    feat_dir = cfg.METHOD.OURS.FEATURE_DIR
    pos_dir = cfg.DATASET.POSITIVE_DIR
    neg_dir = cfg.DATASET.NEGATIVE_DIR
    # image params
    global im_size
    im_size = cfg.TRAIN.SAMPLE_SIZE[cfg.TRAIN.SCHEME]
    # train params
    global bow_ncmps
    bow_ncmps = cfg.METHOD.OURS.BOW_NCOMPONENTS
    print '[Train Ours] Load global configure done.'
    
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
    print '[Train Ours] Clean model directory done.'
            
def shuffle_data(X, Y):
    index = range(len(X))
    random.shuffle(index)
    X = [X[i] for i in index]
    Y = [Y[i] for i in index]
    print '[Train Ours] Shuffle data done.'
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
    print '[Train Ours] Load data done.'
    return X, Y

# opencv svm
def train_svm(X, Y, datatype):
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
    svm.save(os.path.join(model_dir,'svm_'+datatype+'.xml')) 
    # test
    _, pred = svm.predict(X_test)
    right = 0
    for idx, result in enumerate(pred):
        if int(result[0]) == int(Y_test[idx]):
            right += 1
    print '[Train Ours] Linear ' + datatype + ' SVM acc:', right / Y_test.shape[0]
    print '[Train Ours] Linear ' + datatype + ' SVM train done.'
    
# sklearn random forests
def train_rf(X, Y, datatype):
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
    joblib.dump(clf, os.path.join(model_dir,datatype+'_rf.pkl')) 
    # test
    sc = clf.score(X_test,Y_test)
    print '[Train Ours] Random '+datatype+' Forests acc:', sc
    print '[Train Ours] Random '+datatype+' Forests train done.'


def run():
    # load global configue
    load_config()
    # clean model directory
    clean_dir(clean=True)
    # load data from image folders
    X, Y = load_data()
    # shuffle data
    X, Y = shuffle_data(X, Y)

    # extract hog features and train svm(opencv)
    hog_feats = extfeat.extract_hog_from_imlist(X)
    hog_labels = np.array(Y, dtype='int').reshape(-1,1)
    joblib.dump([hog_feats, hog_labels, X], os.path.join(feat_dir,'hog_feats.pkl'))
    train_svm(hog_feats, hog_labels, datatype='HoG')
    
    # extract sift feature and train svm(sklearn)
    sift_feats, _, sift_index = extfeat.extract_sift_from_imlist(X)
    X_bof = extfeat.generate_bof(sift_feats,retrain=True,n_clusters=bow_ncmps,suffix='SIFT')
    SIFT_data = [X[i] for i in sift_index]
    Y_sift = np.array([Y[i] for i in sift_index], dtype='int')
    joblib.dump([X_bof, Y_sift, SIFT_data], os.path.join(feat_dir,'sift_feats.pkl'))
    train_rf(X_bof, Y_sift, datatype='SIFT')
    
    # extract rgbsift feature and train svm(sklearn)
    rgbsift_des, _, rgbsift_ind = extfeat.extract_rgbsift_from_imlist(X)
    rgbsift_b = [elem[0] for elem in rgbsift_des]
    rgbsift_b_bof = extfeat.generate_bof(rgbsift_b,retrain=True,n_clusters=bow_ncmps,suffix='RGBSIFT_B')
    rgbsift_g = [elem[1] for elem in rgbsift_des]
    rgbsift_g_bof = extfeat.generate_bof(rgbsift_g,retrain=True,n_clusters=bow_ncmps,suffix='RGBSIFT_G')
    rgbsift_r = [elem[2] for elem in rgbsift_des]
    rgbsift_r_bof = extfeat.generate_bof(rgbsift_r,retrain=True,n_clusters=bow_ncmps,suffix='RGBSIFT_R')
    rgbsift_bof = np.concatenate((rgbsift_r_bof, rgbsift_g_bof, rgbsift_b_bof), 1)   
    RGBSIFT_data = [X[i] for i in rgbsift_ind]
    Y_rgbsift = np.array([Y[i] for i in rgbsift_ind],dtype='int')
    joblib.dump([rgbsift_bof, Y_rgbsift, RGBSIFT_data], os.path.join(feat_dir, 'rgbsift_feats.pkl'))
    train_rf(rgbsift_bof, Y_rgbsift, datatype='RGBSIFT')

def generate_tsv_data():
    hog_feats, hog_labels, X = joblib.load(os.path.join(feat_dir,'hog_feats.pkl'))
    print hog_feats.shape
    #f_data = open('hog_data.tsv','w')
    #f_meta = open('hog_metadata.tsv','w')
if __name__ == '__main__':
    #run()
    #visualize()
    generate_tsv_data()