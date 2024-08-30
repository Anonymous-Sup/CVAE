# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle

class SKSF_A(BaseImageDataset):
    """
    SKSF_A
    Reference:
    URL: 

    Dataset statistics:
    # identities: 134 
    # images: 134*2 (train) + 134*5 (query) + 134*1 rgb (gallery)
    """
    dataset_dir = 'SKSF-A'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(SKSF_A, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_sketch_dir = osp.join(self.dataset_dir, 'fewshot', '2sketch', 'finetune')
        self.query_sketch_dir = osp.join(self.dataset_dir, 'fewshot', '2sketch', 'test')
        
        rgb_type = 'rgb_all' 

        self.train_rgb_dir = osp.join(self.dataset_dir, 'fewshot', rgb_type)
        # for few-shot setting, train_rgb = gallery_rgb
        self.gallery_rgb_dir = osp.join(self.dataset_dir, 'fewshot', rgb_type)

        self._check_before_run()
        self.pid_begin = pid_begin
        
        train = self._process_train_dir(self.train_rgb_dir, self.train_sketch_dir, relabel=True)
        # train = self._process_train_all_dir(self.train_rgb_dir, self.train_sketch_dir, self.query_sketch_dir, relabel=True)
        query = self._process_query_dir(self.query_sketch_dir, relabel=True)
        gallery = self._process_dir(self.gallery_rgb_dir, relabel=True)

        if verbose:
            print("=> SKSF-A dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_sketch_dir):
            raise RuntimeError("'{}' is not available".format(self.train_sketch_dir))
        if not osp.exists(self.query_sketch_dir):
            raise RuntimeError("'{}' is not available".format(self.query_sketch_dir))
        if not osp.exists(self.train_rgb_dir):
            raise RuntimeError("'{}' is not available".format(self.train_rgb_dir))
        if not osp.exists(self.gallery_rgb_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_rgb_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        # 1.png
        pattern = re.compile(r'(\d+)\.png$')  # Matches digits followed by .png at the end of the string
        
        pid_container = set()
        for img_path in sorted(img_paths):
            pid = int(pattern.search(img_path).group(1))
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid = int(pattern.search(img_path).group(1))
            if pid == -1: continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]
            camid = 0 
            dataset.append((img_path, self.pid_begin + pid, camid, 'rgb'))
        return dataset


    def _process_train_all_dir(self, dir_rgb_path, dir_sketch_path, dir_query_path, relabel=False):
        
        print("===============Remind!!!!============= Using all data for training")
        rgb_img_paths = glob.glob(osp.join(dir_rgb_path, '*.png'))
        rgb_pattern = re.compile(r'(\d+)\.png$')  # Matches digits followed by .png at the end of the string
        
        sketch_img_paths = glob.glob(osp.join(dir_sketch_path, '*.jpg'))
        sketch_img_paths_2 = glob.glob(osp.join(dir_query_path, '*.jpg'))
        sketch_img_paths += sketch_img_paths_2

        # sketch_pattern is like 0001_A.jpg or 0002_B, get the str before and after '_'
        # sketch_pattern = re.compile(r'([-\d]+)_([A-Z])')
        
        # sketch_pattern is like 0001_3.jpg, get the str before and after '_'
        sketch_pattern = re.compile(r'(\d+)_([\d]+)')
        
        # print("Sketch only")
        # rgb_img_paths = []

        # print("RGB only")
        # sketch_img_paths = []

        pid_container = set()
        style_container = set()
        for img_path in sorted(rgb_img_paths):
            pid = int(rgb_pattern.search(img_path).group(1))
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)

        for sketch_path in sorted(sketch_img_paths):
            pid, style_id = map(int, sketch_pattern.search(sketch_path).groups())
            if pid == -1: continue
            style_container.add(style_id)
            # assert pid in pid_container, "sketch {} not in rgb set".format(sketch_path)
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        styleid2label = {style: label for label, style in enumerate(style_container)}

        dataset = []
        for img_path in sorted(rgb_img_paths):
            pid = int(rgb_pattern.search(img_path).group(1))
            if pid == -1: continue  # junk images are just ignored
            camid = 0 # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 'rgb'))

        for sketch_img_path in sorted(sketch_img_paths):
            pid, style_id = map(int, sketch_pattern.search(sketch_path).groups())
            if pid == -1: continue
            if relabel: pid = pid2label[pid]
            style_id = styleid2label[style_id]
            
            # camid viewid are set to 0
            dataset.append((sketch_img_path, self.pid_begin + pid, 0, 'sketch'))

        return dataset
    

    def _process_train_dir(self, dir_rgb_path, dir_sketch_path, relabel=False):
        
        rgb_img_paths = glob.glob(osp.join(dir_rgb_path, '*.png'))
        rgb_pattern = re.compile(r'(\d+)\.png$')  # Matches digits followed by .png at the end of the string

        
        sketch_img_paths = glob.glob(osp.join(dir_sketch_path, '*.jpg'))
        sketch_pattern = re.compile(r'(\d+)_([\d]+)')
        
        pid_container = set()
        style_container = set()

        for img_path in sorted(rgb_img_paths):
            pid = int(rgb_pattern.search(img_path).group(1))
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)

        for sketch_path in sorted(sketch_img_paths):
            pid, style_id = map(int, sketch_pattern.search(sketch_path).groups())
            if pid == -1: continue
            style_container.add(style_id)
            assert pid in pid_container, "sketch {} not in rgb set {}".format(sketch_path, pid_container)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        styleid2label = {style: label for label, style in enumerate(style_container)}

        dataset = []
        for img_path in sorted(rgb_img_paths):
            pid = int(rgb_pattern.search(img_path).group(1))
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= pid <= 134
            # assert 1 <= camid <= 6
            camid = 0  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 'rgb'))

        for sketch_path in sorted(sketch_img_paths):
            pid, style_id = map(int, sketch_pattern.search(sketch_path).groups())
            if pid == -1: continue
            if relabel: pid = pid2label[pid]
            style_id = styleid2label[style_id]
            # camid viewid are set to 0
            dataset.append((sketch_path, self.pid_begin + pid, 0, 'sketch'))
        return dataset
    
    def _process_query_dir(self, dir_sketch_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_sketch_path, '*.jpg'))
        pattern = re.compile(r'(\d+)_([\d]+)')

        pid_container = set()
        style_container = set()
        for img_path in sorted(img_paths):
            pid, style_id = map(int, pattern.search(img_path).groups())
            if pid == -1: continue
            pid_container.add(pid)
            style_container.add(style_id)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        styleid2label = {style: label for label, style in enumerate(style_container)}

        dataset = []
        for img_path in sorted(img_paths):
            pid, style_id = map(int, pattern.search(img_path).groups())
            if pid == -1: continue
            if relabel: pid = pid2label[pid]
            style_id = styleid2label[style_id]
            # camid viewid are set to 0
            dataset.append((img_path, self.pid_begin + pid, 0, 'sketch'))
        return dataset


if __name__== '__main__':
    import sys
    sys.path.append('../')
    market_sketch = SKSF_A(root="/home/zhengwei/Desktop/Zhengwei/Projects/datasets")