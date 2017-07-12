# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
import skimage.io
import skimage.color
from xml.dom.minidom import Document

# 设置的导入图片的路径和标注xml存放路径，可在config.py里面修改
imgfilepath = '/home/yann/Desktop/data/train/'
xmlfilepath = '/home/yann/Desktop/data/annotations/'

# 需要调整参数: 导入的图片后缀，一定要一样
formation = '.jpg'
# 需要调整参数: 图片中水杯数量，人工设定
ballNums = 1
# 需要调整参数: 导入图片名称，不带后缀
imgindex = '73'
xmin = [287,110]
ymin = [190,620]
width = [48,290] #这两个参数指示标定框的款和高
height = [48,290]

# 读取图片
im = skimage.io.imread(imgfilepath+imgindex+formation)
#im2 = skimage.color.rgb2gray(im)
bndboxcolor = ['red','blue','green']
plt.close()
fig,ax = plt.subplots()
ax.imshow(im)
for i in xrange(ballNums):
    ax.add_patch(
        #　图片显示标定框
        plt.Rectangle((xmin[i],ymin[i]),width[i],height[i],
            fill = False,
            edgecolor = bndboxcolor[i],
            linewidth = 2)
    )

fig.show()


#################################################################################################
# 数据写入xml文件保存，不需要关注
doc = Document()
rootNode = doc.createElement('annotation')
doc.appendChild(rootNode)

# 创建文件节点
fileNode = doc.createElement('filename')
fileNode_text = doc.createTextNode(imgindex+formation)
rootNode.appendChild(fileNode)
fileNode.appendChild(fileNode_text)
for i in xrange(ballNums):
    # 创建物体节点
    objectNode = doc.createElement('object')
    # 物体名字，默认cup
    nameNode = doc.createElement('name')
    nameNode_text = doc.createTextNode('cup')
    objectNode.appendChild(nameNode)
    nameNode.appendChild(nameNode_text)
    # 穿件注释节点
    noteNode = doc.createElement('note')
    noteNode_text = doc.createTextNode('x describes the width, y decribes the height')
    objectNode.appendChild(noteNode)
    noteNode.appendChild(noteNode_text)
    # 创建标定框
    bndboxNode = doc.createElement('bndbox')

    xminNode = doc.createElement('xmin')
    xminNode_text = doc.createTextNode(str(xmin[i]))
    bndboxNode.appendChild(xminNode)
    xminNode.appendChild(xminNode_text)

    yminNode = doc.createElement('ymin')
    yminNode_text = doc.createTextNode(str(ymin[i]))
    bndboxNode.appendChild(yminNode)
    yminNode.appendChild(yminNode_text)

    xmaxNode = doc.createElement('xmax')
    xmaxNode_text = doc.createTextNode(str(xmin[i]+width[i]))
    bndboxNode.appendChild(xmaxNode)
    xmaxNode.appendChild(xmaxNode_text)

    ymaxNode = doc.createElement('ymax')
    ymaxNode_text = doc.createTextNode(str(ymin[i]+height[i]))
    bndboxNode.appendChild(ymaxNode)
    ymaxNode.appendChild(ymaxNode_text)
    objectNode.appendChild(bndboxNode)
    rootNode.appendChild(objectNode)

if not os.path.exists(xmlfilepath):
    os.makedirs(xmlfilepath)
    
f = open(xmlfilepath+imgindex+'.xml','w')
f.write(doc.toprettyxml(indent=''))
f.close()