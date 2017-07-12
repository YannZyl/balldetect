# -*- coding: utf-8 -*-
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add root to PYTHONPATH
root_path = osp.join(this_dir, '..')
add_path(root_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

# Add spatial pyramid match direectory to PYTHONPATH
spm_path = osp.join(this_dir, '..', 'spatial_pyramid_match')
spm_recg_path = osp.join(this_dir, '..', 'spatial_pyramid_match','recognition')
add_path(spm_path)
add_path(spm_recg_path)

# Add selective search direectory to PYTHONPATH
ss_path = osp.join(this_dir, '..', 'selective_search')
add_path(ss_path)