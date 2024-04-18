# encoding: utf-8
import glob
import re

import os.path as osp

from collections import Counter
import pickle
from scipy.io import loadmat
import torch

class MarketSketch(object):
    """
    Market-Sketch-1K
    Reference:
    Lin et al. ACM MM 2023
    URL: 

    Dataset statistics:
    # identities: 996 
    # images: 996*3 (train) + 996*2 (query) + others (gallery)
    """
    root_folder = 'Market-Sketch-1K'

    def __init__(self, root='root', format_tag='tensor',  pretrained='CLIPreidNew', latent_size=12, test_metrix_only=False, pid_begin = 0, **kwargs):
        super(MarketSketch, self).__init__()
        
        self.tag = format_tag
        self.test_metrix_only = test_metrix_only
        self.latent_size = latent_size

        if self.tag == 'tensor':
            self.dataset_dir = osp.join(root, self.root_folder, 'tensor', pretrained)
        else:
            self.dataset_dir = osp.join(root, self.root_folder)
        
        self.train_sketch_dir = osp.join(self.dataset_dir, 'sketch', 'fewshot', 'all', 'finetune')
        self.query_sketch_dir = osp.join(self.dataset_dir, 'sketch', 'fewshot', 'all', 'test')
        
        self.train_rgb_dir = osp.join(self.dataset_dir, 'photo', 'all')
        # for few-shot setting, train_rgb = gallery_rgb
        self.gallery_rgb_dir = osp.join(self.dataset_dir, 'photo', 'all')
        self.cluster_dir = osp.join(root, self.root_folder, 'kmeans_results', pretrained)

        self._check_before_run()

        self.pid_begin = pid_begin
        
        train, num_train_pids, num_train_imgs, train_styles, train_centroids = self._process_train_dir(self.train_rgb_dir, self.train_sketch_dir, self.tag, self.latent_size, self.test_metrix_only, relabel=True)
        query, num_query_pids, num_query_imgs, num_query_styles, query_centroids = self._process_query_dir(self.query_sketch_dir, self.tag, self.latent_size, self.test_metrix_only, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs, gallery_centroids = self._process_dir(self.gallery_rgb_dir, self.tag, self.latent_size, self.test_metrix_only, relabel=False)

        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        if self.tag == 'tensor':
            print("=> Market-Sketch-1K tensor loaded")
        else:
            print("=> Market-Sketch-1K loaded")
        
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

        self.train_centroids = train_centroids
        self.query_centroids = query_centroids
        self.gallery_centroids = gallery_centroids

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

    def _process_dir(self, dir_path, tag, latent_size, test_metrix_only, relabel=False):
        
        if tag == 'tensor':
            img_paths = glob.glob(osp.join(dir_path, '*.pt'))
        else:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        pattern = re.compile(r'([-\d]+)_c(\d)')

        if not test_metrix_only:
            cluster_dim = 2*latent_size+1
            _, path2label, centroids = self._process_cluster(dir_path, cluster_dim)
        else:
            centroids = None

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            file_name = osp.basename(img_path)
            file_name = re.sub(r'\.pt$', '', file_name)
            if not test_metrix_only:
                clurster_id = path2label[file_name]
            else:
                clurster_id = 0
            if pid == -1: continue  # junk images are just ignored
            # assert 0 <= pid <= 1501  # pid == 0 means background
            # assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, clurster_id))
        
        num_pids = len(pid_container)
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, centroids
    
    def _process_train_dir(self, dir_rgb_path, dir_sketch_path, tag, latent_size, test_metrix_only, relabel=False):
        
        if tag == 'tensor':
            rgb_img_paths = glob.glob(osp.join(dir_rgb_path, '*.pt'))
            sketch_img_paths = glob.glob(osp.join(dir_sketch_path, '*.pt'))
        else:
            rgb_img_paths = glob.glob(osp.join(dir_rgb_path, '*.jpg'))
            sketch_img_paths = glob.glob(osp.join(dir_sketch_path, '*.jpg'))

        rgb_pattern = re.compile(r'([-\d]+)_c(\d)')
        # sketch_pattern is like 0001_A.jpg or 0002_B, get the str before and after '_'
        sketch_pattern = re.compile(r'([-\d]+)_([A-Z])')
        
        if not test_metrix_only:
            cluster_dim = 2*latent_size+1
            self.cluster_id_begin = cluster_dim - 1
            _, path2label_rgb, centroids_rgb = self._process_cluster(dir_rgb_path, cluster_dim)
            _, path2label_sketch, centroids_sketch = self._process_cluster(dir_sketch_path, cluster_dim)
        else:
            centroids_rgb = None
            centroids_sketch = None
            self.cluster_id_begin = 0
        
        pid_container = set()
        style_container = set()
        for img_path in sorted(rgb_img_paths):
            pid, _ = map(int, rgb_pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)

        for sketch_path in sorted(sketch_img_paths):
            pid, style_id = sketch_pattern.search(sketch_path).groups()
            pid = int(pid)
            if pid == -1: continue
            style_container.add(style_id)
            assert pid in pid_container, "sketch {} not in rgb set".format(sketch_path)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        styleid2label = {style: label for label, style in enumerate(style_container)}

        dataset = []
        for img_path in sorted(rgb_img_paths):
            pid, camid = map(int, rgb_pattern.search(img_path).groups())

            file_name = osp.basename(img_path)
            file_name = re.sub(r'\.pt$', '', file_name)
            if not test_metrix_only:
                clurster_id = path2label_rgb[file_name]
            else:
                clurster_id = 0

            if pid == -1: continue  # junk images are just ignored
            # assert 0 <= pid <= 1501  # pid == 0 means background
            # assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, clurster_id))

        for sketch_img_path in sorted(sketch_img_paths):
            pid, style_id = sketch_pattern.search(sketch_img_path).groups()
            pid = int(pid)
            
            file_name = osp.basename(sketch_img_path)
            file_name = re.sub(r'\.pt$', '', file_name)
            if not test_metrix_only:
                clurster_id = path2label_sketch[file_name] + self.cluster_id_begin
            else:
                clurster_id = 0
            
            if pid == -1: continue
            if relabel: pid = pid2label[pid]
            style_id = styleid2label[style_id]
            
            # camid viewid are set to 0
            dataset.append((sketch_img_path, self.pid_begin + pid, 0, clurster_id))

        num_pids = len(pid_container)
        num_styles = len(style_container)
        num_imgs = len(dataset)
        centroids_merge = torch.cat((centroids_rgb, centroids_sketch), dim=0)
        return dataset, num_pids, num_imgs, num_styles, centroids_merge


    def _process_query_dir(self, dir_sketch_path, tag, latent_size, test_metrix_only, relabel=False):
        
        if tag == 'tensor':
            img_paths = glob.glob(osp.join(dir_sketch_path, '*.pt'))
        else:
            img_paths = glob.glob(osp.join(dir_sketch_path, '*.jpg'))
    
        pattern = re.compile(r'([-\d]+)_([A-Z])')

        if not test_metrix_only:
            cluster_dim = 2*latent_size+1
            _, path2label, centroids= self._process_cluster(dir_sketch_path, cluster_dim)
        else:
            centroids = None

        pid_container = set()
        style_container = set()
        for img_path in sorted(img_paths):
            pid, style_id = pattern.search(img_path).groups()
            pid = int(pid)
            if pid == -1: continue
            pid_container.add(pid)
            style_container.add(style_id)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        styleid2label = {style: label for label, style in enumerate(style_container)}

        dataset = []
        for img_path in sorted(img_paths):
            pid, style_id = pattern.search(img_path).groups()
            pid = int(pid)

            file_name = osp.basename(img_path)
            file_name = re.sub(r'\.pt$', '', file_name)
            if not test_metrix_only:
                clurster_id = path2label[file_name]
            else:
                clurster_id = 0

            if pid == -1: continue
            if relabel: pid = pid2label[pid]
            style_id = styleid2label[style_id]
            # camid viewid are set to 0
            dataset.append((img_path, self.pid_begin + pid, 0, clurster_id))
        
        num_pids = len(pid_container)
        num_styles = len(style_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs, num_styles, centroids
    
    def _process_cluster(self, dir_path, n_clusters=25, sim_mode='euclidean'):

        # n_clusters selected bt 12*2+1 = 25 or 36*2+1 = 73 or 64*2+1 = 129
        if 'photo' in dir_path:
            # both rgb and sketch 129c+129c?
            folder_type = 'train_rgb'
        elif 'finetune' in dir_path:
            folder_type = 'train_sketch'
        elif 'test' in dir_path:
            # sketch only 129c
            folder_type = 'query'
        else:
            raise RuntimeError("Unkown folder type")

        kmeans_file = osp.join(self.cluster_dir, '{}_{}k_{}.pt'.format(folder_type, n_clusters, sim_mode))
        print("Loading cluster results from '{}'".format(kmeans_file))
        # load .pt files
        cluster_resutls = torch.load(kmeans_file)
        labels = cluster_resutls['labels']
        path2label = cluster_resutls['path2label']
        centroids = cluster_resutls['centroids']

        # distroy cluster_resutls
        del cluster_resutls

        counter = Counter(labels.cpu().numpy())
        sorted_counter = sorted(counter.items(), key=lambda x: x[0])  # sort by element
        print("Clustering stastistics:", sorted_counter)
        return labels, path2label, centroids


if __name__== '__main__':
#     import sys
#     sys.path.append('../')
#     market_sketch = MarketSketch(root="/home/zhengwei/Desktop/Zhengwei/Projects/datasets")
    root = "/home/zhengwei/Desktop/Zhengwei/Projects/datasets"
    dataset = MarketSketch(root=root, format_tag='tensor', pretrained='CLIPreidNew', latent_size=64, test_metrix_only=False)
    print(dataset.train_centroids.size())