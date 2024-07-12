# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import os.path as osp
from collections import defaultdict
from .bases import BaseImageDataset
import pickle

class SYSU_MM01(BaseImageDataset):
    """
    SYSU_MM01-Fewshot setting

    Dataset statistics:
    # identities: 451 
    # Training images: 451 * 1 * 4cam (RGB) + 451 * 2 * 2 cam (infrared) 
      Testing images others
    """
    dataset_dir = 'sysu_mm_01'
    def __init__(self, root='root', format_tag='tensor', pretrained='CLIPreidFinetune', pid_begin = 0, **kwargs):
        super(SYSU_MM01, self).__init__()

        self.tag = format_tag

        if self.tag == 'tensor':
            self.dataset_dir = osp.join(root, self.dataset_dir, 'fewshot_label', 'tensor', pretrained)
            suffix = '*.pt'
        else:
            self.dataset_dir = osp.join(root, self.dataset_dir, 'fewshot_label')
            suffix = '*.jpg'

        self.train_infrared_dir = osp.join(self.dataset_dir, '2infrared', 'finetune')
        
        self.query_infrared_dir = osp.join(self.dataset_dir, '2infrared', 'test')
        
        # rgb_type = 'gaussian_all'   # 'b+all', 'b-all', 'all', 'gaussian_all'

        self.train_rgb_dir = osp.join(self.dataset_dir, 'visiable', 'finetune')
        # for few-shot setting, train_rgb = gallery_rgb
        self.gallery_rgb_dir = osp.join(self.dataset_dir, 'visiable', 'test')

        self._check_before_run()
        self.pid_begin = pid_begin
        
        train = self._process_train_dir(self.train_rgb_dir, self.train_infrared_dir, relabel=True, suffix=suffix)
        # train = self._process_train_all_dir(self.train_rgb_dir, self.train_infrared_dir, self.query_infrared_dir, self.gallery_rgb_dir, relabel=True)
        query = self._process_dir(self.query_infrared_dir, relabel=True, data_tag='infrared', suffix=suffix)
        gallery = self._process_dir(self.gallery_rgb_dir, relabel=True, data_tag='rgb', suffix=suffix)

        if self.tag == 'tensor':
            print("=> SYSU_MM01 tensor loaded")
        else:
            print("=> SYSU_MM01 loaded")
        
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
        if not osp.exists(self.train_infrared_dir):
            raise RuntimeError("'{}' is not available".format(self.train_infrared_dir))
        if not osp.exists(self.query_infrared_dir):
            raise RuntimeError("'{}' is not available".format(self.query_infrared_dir))
        if not osp.exists(self.train_rgb_dir):
            raise RuntimeError("'{}' is not available".format(self.train_rgb_dir))
        if not osp.exists(self.gallery_rgb_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_rgb_dir))

    def _process_dir(self, dir_path, relabel=False, data_tag='rgb', suffix='*.pt'):
        img_paths = glob.glob(osp.join(dir_path, suffix))
        # name = 0001_0001_c2.jpg, get the str before and after '_'
        pattern = re.compile(r'(\d+)_(\d+)_c(\d+)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _, camid= map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, _, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            # assert 0 <= pid <= 1501  # pid == 0 means background
            # assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, data_tag))
        return dataset
    

    def _process_train_all_dir(self, dir_rgb_path, dir_infrared_path, dir_query_path, dir_gallery_path, relabel=False, suffix='*.pt'):
        
        print("===============Remind!!!!============= Using all data for training")
        rgb_img_paths = glob.glob(osp.join(dir_rgb_path, suffix))
        rgb_img_paths2 = glob.glob(osp.join(dir_gallery_path, suffix))
        rgb_img_paths += rgb_img_paths2
        
        infrared_img_paths = glob.glob(osp.join(dir_infrared_path, suffix))
        infrared_img_paths2 = glob.glob(osp.join(dir_query_path, suffix))
        infrared_img_paths += infrared_img_paths2

        pattern = re.compile(r'(\d+)_(\d+)_c(\d+)')

        pid_container = set()
        for img_path in sorted(rgb_img_paths):
            pid, _, _= map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)

        for infrared_path in sorted(infrared_img_paths):
            pid, _, camid = pattern.search(infrared_path).groups()
            pid = int(pid)
            if pid == -1: continue
            # assert pid in pid_container, "sketch {} not in rgb set".format(sketch_path)
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in sorted(rgb_img_paths):
            pid, _, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            # assert 0 <= pid <= 1501  # pid == 0 means background
            # assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 'rgb'))

        for infrared_path in sorted(infrared_img_paths):
            pid, _, camid = map(int, pattern.search(infrared_path).groups())
            pid = int(pid)
            if pid == -1: continue
            if relabel: pid = pid2label[pid]
            camid -= 1  # index starts from 0
            dataset.append((infrared_path, self.pid_begin + pid, camid, 'infrared'))

        return dataset
    

    def _process_train_dir(self, dir_rgb_path, dir_infrared_path, relabel=False, suffix='*.pt'):
        rgb_img_paths = glob.glob(osp.join(dir_rgb_path, suffix))
        pattern = re.compile(r'(\d+)_(\d+)_c(\d+)')
        
        intrared_img_paths = glob.glob(osp.join(dir_infrared_path, suffix))

        pid_container = set()
  
        for img_path in sorted(rgb_img_paths):
            pid, _, _= map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)

        for infrared_img_path in sorted(intrared_img_paths):
            pid, _, camid = map(int, pattern.search(infrared_img_path).groups())
            pid = int(pid)
            if pid == -1: continue
            assert pid in pid_container, "infrared {} not in rgb set".format(infrared_img_path)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in sorted(rgb_img_paths):
            pid, _, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            # assert 0 <= pid <= 1501  # pid == 0 means background
            # assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 'rgb'))

        for infrared_img_path in sorted(intrared_img_paths):
            pid, _, camid = map(int, pattern.search(infrared_img_path).groups())
            pid = int(pid)
            if pid == -1: continue
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((infrared_img_path, self.pid_begin + pid, camid, 'infrared'))

        return dataset
    

if __name__== '__main__':
#     import sys
#     sys.path.append('../')
    root = "/home/zhengwei/Desktop/Zhengwei/Projects/datasets"
    dataset = SYSU_MM01(root=root, format_tag='tensor', pretrained='CLIPreidFinetune')
    print("ok")