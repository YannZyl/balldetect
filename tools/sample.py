# -*- coding: utf-8 -*-
from __future__ import division
import init_path
import os
from skimage import io,transform
from xml.dom import minidom
from config import cfg

# dataset path
bkgPath = cfg.DATASET.BACKGROUND_DIR
xmlPath = cfg.DATASET.ANNOTATION_DIR
imgPath = cfg.DATASET.IMAGES_DIR
posPath = cfg.DATASET.POSITIVE_DIR
negPath = cfg.DATASET.NEGATIVE_DIR
# sampling scale
overlap_pos = cfg.SAMPLING.POS_OVERLAP
overlap_neg = cfg.SAMPLING.NEG_OVERLAP
# output background and spatial pyramid scale
outputscale = cfg.SAMPLING.OUTPUT_SCALE
bkgscale = cfg.SAMPLING.BACKGROUND_SCALE
spscale = cfg.SAMPLING.SCALE
# save image index
image_index = 100000
    
def multiScale(imgpath,image):
    global outputscale, spscale, image_index
    for sc in spscale:
        img = transform.rescale(image,sc);
        img = transform.resize(img,outputscale)
        io.imsave(os.path.join(imgpath,str(image_index)+'.jpg'),img)  
        image_index += 1    
    
# generate function
def generateSamples(image,gt_box,overlap,imgpath):    
    global outputscale
    [xmin,ymin,xmax,ymax] = gt_box
    imgshape = image.shape    
    # check bbox size
    if xmax - xmin < 10 or ymax - ymin < 10:
        return 
    # loop for each scale
    for over in overlap:      
        # generate though y-axis direction
        if imgshape[0] - ymax > ymin:
            startRow = int(ymin + (ymax - ymin) * (1 - over))
        else:
            startRow = int(ymin - (ymax - ymin) * (1 - over))
            
        if startRow >= 0 and startRow+(ymax-ymin) < imgshape[0]:
            img1 = image[startRow:startRow+(ymax-ymin),xmin:xmax,:]   
            img1 = transform.resize(img1,outputscale) 
            multiScale(imgpath,img1)
            
        # generate though x-axis direction
        if imgshape[1] - xmax > xmin:
            startCol = int(xmin + (xmax - xmin) * (1 - over))
        else:
            startCol = int(xmin - (xmax - xmin) * (1 - over))
            
        if startCol >=0 and startCol+(xmax-xmin) < imgshape[0]:
            img2 = image[ymin:ymax,startCol:startCol+(xmax-xmin),:] 
            img2 = transform.resize(img2,outputscale)   
            multiScale(imgpath,img2)
           
# read xml file and generate samples
def samplesGenerator():   
    global bkgPath, xmlPath, imgPath, posPath, negPath 
    global overlap_pos, overlap_neg
    for xmlname in os.listdir(xmlPath):
        # load xml
        dom = minidom.parse(os.path.join(xmlPath,xmlname))
        root = dom.documentElement
        imgname = root.getElementsByTagName('filename')[0].firstChild.data
        obj_xmin = root.getElementsByTagName('xmin')
        obj_ymin = root.getElementsByTagName('ymin')
        obj_xmax = root.getElementsByTagName('xmax')
        obj_ymax = root.getElementsByTagName('ymax')
        # read image
        image = io.imread(os.path.join(imgPath,imgname)) 
        for index in xrange(len(obj_xmin)):
            xmin = int(obj_xmin[index].firstChild.data)
            ymin = int(obj_ymin[index].firstChild.data)
            xmax = int(obj_xmax[index].firstChild.data)
            ymax = int(obj_ymax[index].firstChild.data)
            # groundtruth boundingbox
            gt_box = [xmin,ymin,xmax,ymax]
            # generate positive samples
            generateSamples(image,gt_box,overlap_pos,posPath)
            # generate negative samples
            generateSamples(image,gt_box,overlap_neg,negPath)
            
# generate positive samples utilize background image
def generateBackgroundSamples():
    global bkgPath, outputscale, outputscale
    # sampling stride
    stride = 20
    # read background image and generation
    for imgname in os.listdir(bkgPath):
        img = io.imread(os.path.join(bkgPath,imgname))
        # different scale
        for scale in bkgscale:
            print img.shape, scale
            img_cpy = transform.rescale(img,scale)
            print img.shape, scale, img_cpy.shape
            # save samples
            for i in xrange(0,img_cpy.shape[0]-outputscale[1],stride):
                for j in xrange(0,img_cpy.shape[1]-outputscale[0],stride):
                    block = img_cpy[i:i+outputscale[1],j:j+outputscale[0],:]
                    if block.shape[0] != outputscale[1] or block.shape[1] != outputscale[0]:
                        continue
                    multiScale(negPath,block)      

# 生成样本txt文件列表
def generateTxtLists():
    # 生成正样本名称列表
    posf = open('positive_samples_lists.txt','w')
    for file in os.listdir(posPath):
        posf.writelines(file+'\n')
    posf.close()
    
    # 生成负样本名称列表
    negf = open('negative_samples_lists.txt','w')
    for file in os.listdir(negPath):
        negf.writelines(file+'\n')
    negf.close()

# clean function                 
def cleanDir():
    if not os.path.exists(posPath):
        os.makedirs(posPath)
    else:
        for item in os.listdir(posPath):
            os.remove(os.path.join(posPath,item)) 
        
    if not os.path.exists(negPath):
        os.makedirs(negPath)
    else:
        for item in os.listdir(negPath):
            os.remove(os.path.join(negPath,item))   

# main function
if __name__ == '__main__':
    # clean directory
    cleanDir() 
    # generate positive and negative samples utilize xml file
    samplesGenerator()
    # genrate negative samples utilize background image
    generateBackgroundSamples()
    # generate image list
    # generateTxtLists()
