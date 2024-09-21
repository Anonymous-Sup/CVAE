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
import random

class MarketSketch_FewShot(BaseImageDataset):
    """
    Market-Sketch-1K
    Reference:
    Lin et al. ACM MM 2023
    URL: 

    Dataset statistics:
    # identities: 996 

    """
    root_folder = 'Market-Sketch-1K'
    def __init__(self, root='', format_tag='tensor', pretrained='CLIPreidFinetune', pid_begin=0, NWAY=5, KSHOT=1, **kwargs):
        super(MarketSketch_FewShot, self).__init__()
        
        self.training_mode = 'novel'
        self.tag = format_tag
        self.pid_begin = pid_begin
        self.NWAY = NWAY
        self.KSHOT = KSHOT

        if self.KSHOT == 1:
            folder = 'all'
        elif self.KSHOT == 2:
            folder = '2sketch'

        if self.tag == 'tensor':
            self.dataset_dir = osp.join(root, self.root_folder, 'tensor', pretrained)
        else:  
            self.dataset_dir = osp.join(root, self.root_folder)

        self.sketch_dir = osp.join(self.dataset_dir, 'sketch', 'fewshot', folder, 'finetune')
        self.sketch_dir_rest = osp.join(self.dataset_dir, 'sketch', 'fewshot', folder, 'test')
        
        rgb_type = 'orgin_all'   # 'b+all', 'b-all', 'all', 'gaussian_all'

        self.rgb_selected = osp.join(self.dataset_dir, 'photo', 'all')
        self.rgb_dir = osp.join(self.dataset_dir, 'photo', rgb_type)

        self._check_before_run()
        self.pid_begin = pid_begin
        
        train, val, query, gallery = self._process_dir(self.rgb_dir, self.rgb_selected, self.sketch_dir, self.sketch_dir_rest, relabel=True, 
                                                       training_mode=self.training_mode, random_seed=0,
                                                       number_pthots=self.KSHOT, number_sketches=self.KSHOT)
        
        print("=> Market-Sketch-1K Few-shot loaded")
        self.print_dataset_statistics(train, query, gallery, val)

        self.train = train
        self.val = val
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.sketch_dir):
            raise RuntimeError("'{}' is not available".format(self.sketch_dir))
        if not osp.exists(self.sketch_dir_rest):
            raise RuntimeError("'{}' is not available".format(self.sketch_dir_rest))
        if not osp.exists(self.rgb_dir):
            raise RuntimeError("'{}' is not available".format(self.rgb_dir))
    
    def _process_dir(self, rgb_path, rgb_sub_path, sketch_path, sketch_path_rest, relabel=False, training_mode='novel', number_pthots=1, number_sketches=1, eposido=100, number_way=5, random_seed=0):
        
        Way = number_way
        N = number_pthots
        M = number_sketches

        # Set the random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)

        rgb_img_paths = glob.glob(osp.join(rgb_path, '*.jpg'))
        rgb_sub_img_paths = glob.glob(osp.join(rgb_sub_path, '*.jpg'))


        sketch_img_paths = glob.glob(osp.join(sketch_path, '*.jpg'))
        sketch_img_paths_2 = glob.glob(osp.join(sketch_path_rest, '*.jpg'))
        sketch_img_paths += sketch_img_paths_2
        # sketch_pattern is like 0001_A.jpg or 0002_B, get the str before and after '_'
        
        rgb_pattern = re.compile(r'([-\d]+)_c(\d)')
        sketch_pattern = re.compile(r'([-\d]+)_([A-Z])')

        pid_container = set()
        style_container = set()

        for img_path in sorted(rgb_img_paths):
            pid, _ = map(int, rgb_pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for sketch_path in sorted(sketch_img_paths):
            pid, style_id = sketch_pattern.search(sketch_path).groups()
            pid = int(pid)
            if pid == -1: continue
            style_container.add(style_id)
            assert pid in pid_container, "sketch {} not in rgb set".format(sketch_path)
            pid_container.add(pid)
        styleid2label = {style: label for label, style in enumerate(style_container)}
        
        if len(pid_container) < Way:
            raise ValueError("Not enough classes available to sample {} classes.".format(Way))


        # Datasets to hold the few-shot data
        train_dataset = []
        query_dataset = []
        gallery_dataset = []
        # Sample classes for the few-shot episode
        
        if training_mode == 'novel':
            sampled_pids = pid_container
        elif training_mode == 'novel_few':
            # Sample classes for the few-shot episode
            sampled_pids = random.sample(pid_container, Way)
            # print("sampled_pids", sampled_pids)
            # rebuild few_pid2label
            pid2label = {pid: label for label, pid in enumerate(sampled_pids)}


        for pid in sampled_pids:
            # Get all RGB images for this class
            class_rgb_images = [img_path for img_path in rgb_sub_img_paths if int(rgb_pattern.search(img_path).groups()[0]) == pid]
            class_rgb_images_basenames = []
            class_all_rgb_images = [img_path for img_path in rgb_img_paths if int(rgb_pattern.search(img_path).groups()[0]) == pid]
            
            # Get all sketches for this class
            class_sketch_images = [sketch_path for sketch_path in sketch_img_paths if int(sketch_pattern.search(sketch_path).groups()[0]) == pid]
            
            # Check if there are enough samples for few-shot learning
            # images in sub is maximax 2
            if len(class_rgb_images) < N or len(class_sketch_images) < M:
                raise ValueError(f"Class {pid} does not have enough samples for the few-shot setting.")
            
            # Randomly select N RGB images for training
            selected_rgb_images = random.sample(class_rgb_images, N)
            # Randomly select M sketch images for training
            selected_sketch_images = random.sample(class_sketch_images, M)

            # Add selected images to the training dataset
            for img_path in selected_rgb_images:
                class_rgb_images_basenames.append(osp.basename(img_path))
                _, camid = map(int, rgb_pattern.search(img_path).groups())
                camid -= 1  # Camid starts from 0
                if relabel:
                    new_pid = pid2label[pid]
                train_dataset.append((img_path, self.pid_begin + new_pid, camid, 'rgb'))
            
            for sketch_path in selected_sketch_images:
                _, style_id = sketch_pattern.search(sketch_path).groups()
                if relabel:
                    new_pid = pid2label[pid]
                # camid/viewid set to 0
                train_dataset.append((sketch_path, self.pid_begin + new_pid, 0, 'sketch'))

            # Remaining RGB images are used for gallery
            remaining_rgb_images = [img_path for img_path in class_all_rgb_images if osp.basename(img_path) not in class_rgb_images_basenames]
            for img_path in remaining_rgb_images:
                _, camid = map(int, rgb_pattern.search(img_path).groups())
                camid -= 1  # Camid starts from 0
                if relabel:
                    new_pid = pid2label[pid]
                gallery_dataset.append((img_path, self.pid_begin + new_pid, camid, 'rgb'))

            # Remaining sketches are used for query
            remaining_sketch_images = [sketch_path for sketch_path in class_sketch_images if sketch_path not in selected_sketch_images]
            for sketch_path in remaining_sketch_images:
                _, style_id = sketch_pattern.search(sketch_path).groups()
                if relabel:
                    new_pid = pid2label[pid]
                query_dataset.append((sketch_path, self.pid_begin + new_pid, 0, 'sketch'))
            
        # Combine gallery and query datasets for validation
        val_dataset = gallery_dataset + query_dataset

        return train_dataset, val_dataset, query_dataset, gallery_dataset

# if __name__== '__main__':
#     import sys
#     sys.path.append('../')
#     market_sketch = MarketSketch(root="/home/zhengwei/Desktop/Zhengwei/Projects/datasets")