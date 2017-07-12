import numpy as np
from config import cfg
import vocabulary
import os.path as osp
from sklearn.externals import joblib

def spm_fit(spm_info,level=2,nclusters=10):
    sift_des = spm_info['des']
    image_sets = spm_info['img']
    sift_pts = spm_info['kps']
    # construct vocabulary
    voc = vocabulary.Vocabulary(sift_des, nclusters)
    joblib.dump(voc,osp.join(cfg.METHOD.SPM.MODEL_DIR, 'spm_voc.pkl')) 
    
    # compute spm based on given level
    spm_feats = []
    for index in xrange(len(sift_des)): 
        level = 2
        image_info = {'image_shape':image_sets[index].shape, \
                      'image_des':sift_des[index], \
                      'image_kps':sift_pts[index]}
        spm_hist = voc.buildHistogramForEachImageAtDifferentLevels(image_info, level)
        spm_feats.append(spm_hist)
    
    spm_feats = np.array(spm_feats, dtype='float32')
    return spm_feats
    
def spm_transform(spm_info, level=2):
    sift_des = spm_info['des']
    image_sets = spm_info['img']
    sift_pts = spm_info['kps']
    # load vocabulary
    voc = joblib.load(osp.join(cfg.METHOD.SPM.MODEL_DIR, 'spm_voc.pkl'))
    # compute spm based on given level
    spm_feats = []
    for index in xrange(len(sift_des)): 
        level = 2
        image_info = {'image_shape':image_sets[index].shape, \
                      'image_des':sift_des[index], \
                      'image_kps':sift_pts[index]}
        spm_hist = voc.buildHistogramForEachImageAtDifferentLevels(image_info, level)
        spm_feats.append(spm_hist)
    spm_feats = np.array(spm_feats, dtype='float32')
    return spm_feats