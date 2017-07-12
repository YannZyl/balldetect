# -*- coding: utf-8 -*-
import init_path
import os
import detect_ours
from config import cfg
import matplotlib.pyplot as plt

for img_name in os.listdir(cfg.DATASET.TEST_DIR):
    imgpath = os.path.join(cfg.DATASET.TEST_DIR, img_name)
    detect_ours.main_detect(imgpath)

"""
from spm import extspm

feature,label = extspm(image_folder=cfg.DATASET.POSITIVE_DIR, label=1, level=2)
print  len(feature), len(feature[0]),len(label), label[0:10]
"""
"""
import selective_search as ss
from skimage import io
img = io.imread(os.path.join(cfg.DATASET.TEST_DIR, '55.jpg'))
regions = ss.selective_search(img, 
                           color_spaces = ['rgb'],
                           ks = [150],
                           feature_masks = [(0, 0, 1, 1)]) 
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(img, aspect='equal')
for v, (i0, j0, i1, j1) in regions:
    #print bbox
    ax.add_patch(
        plt.Rectangle((j0, i0),
                 j1-j0,i1-i0, fill=False,
                 edgecolor='red', linewidth=1.5)
            )

ax.set_title(('{} detections result').format('ss'),
                  fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.draw()
plt.show()
"""