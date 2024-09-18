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

train_classes_split = [
    "airplane",
    "alarm_clock",
    "ant",
    "ape",
    "apple",
    "armor",
    "axe",
    "banana",
    "bear",
    "bee",
    "beetle",
    "bell",
    "bench",
    "bicycle",
    "blimp",
    "bread",
    "butterfly",
    "camel",
    "candle",
    "cannon",
    "car_(sedan)",
    "castle",
    "cat",
    "chair",
    "chicken",
    "church",
    "couch",
    "crab",
    "crocodilian",
    "cup",
    "deer",
    "dog",
    "duck",
    "elephant",
    "eyeglasses",
    "fan",
    "fish",
    "flower",
    "frog",
    "geyser",
    "guitar",
    "hamburger",
    "hammer",
    "harp",
    "hat",
    "hedgehog",
    "hermit_crab",
    "horse",
    "hot_air_balloon",
    "hotdog",
    "hourglass",
    "jack_o_lantern",
    "jellyfish",
    "kangaroo",
    "knife",
    "lion",
    "lizard",
    "lobster",
    "motorcycle",
    "mushroom",
    "owl",
    "parrot",
    "penguin",
    "piano",
]

val_classes_split = [
    "pickup_truck",
    "pig",
    "pineapple",
    "pistol",
    "pizza",
    "pretzel",
    "rabbit",
    "racket",
    "ray",
    "rifle",
    "rocket",
    "sailboat",
    "saxophone",
    "scorpion",
    "sea_turtle",
    "seal",
    "shark",
    "sheep",
    "shoe",
    "snail",
    "snake",
    "spider",
    "spoon",
    "squirrel",
    "starfish",
    "strawberry",
    "swan",
    "table",
    "tank",
    "teapot",
    "teddy_bear",
    "tiger",
    "trumpet",
    "turtle",
    "umbrella",
    "violin",
    "volcano",
    "wading_bird",
    "wine_bottle",
    "zebra",
]

novel_classes_split = [
    "bat",
    "cabin",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "pear",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "seagull",
    "skyscraper",
    "songbird",
    "sword",
    "tree",
    "wheelchair",
    "windmill",
    "window",
]


def buildLabelIndex(Way=5, seed=0):
    base_labels = train_classes_split 
    novel_labels = novel_classes_split + val_classes_split 
    random.seed(seed)
    selected_labels = random.sample(novel_labels, Way)
    base_label2inds = {}
    novel_label2inds = {}
    selected_label2inds = {}
    for idx, label in enumerate(base_labels):
        # if label not in label2inds:
        #    label2inds[label] = []
        # label2inds[label].append(idx)
        base_label2inds[label] = idx

    for idx, label in enumerate(novel_labels):
        novel_label2inds[label] = idx

    for idx, label in enumerate(selected_labels):
        selected_label2inds[label] = idx
    return base_label2inds, novel_label2inds, selected_label2inds


class Sketchy(BaseImageDataset):
    """
    Sketchy
    Reference:
    URL: 

    Dataset statistics:
    Base64 Dataset statistics:
    ----------------------------------------
    subset   | # ids | # images | # cameras
    ----------------------------------------
    train    |    64 |    23217 |         1
    query    |    64 |    32323 |         1
    gallery  |    64 |     7740 |         1
    val      |    64 |     7739 |         1
    ----------------------------------------

    Novel61 Dataset statistics: 5-shot
    ----------------------------------------
    subset   | # ids | # images | # cameras
    ----------------------------------------
    train    |    61 |      610 |         1
    query    |    61 |    36392 |         1
    gallery  |    61 |    34001 |         1
    val      |    61 |    70393 |         1
    ----------------------------------------     
    """

    root_folder = 'Sketchy'

    def __init__(self, root='', format_tag='tensor', pretrained='CLIPreidFinetune', pid_begin=0, NWAY=5, KSHOT=1, **kwargs):
        super(Sketchy, self).__init__()

        # self.training_mode = config.DATASETS.TRAINING_MODE # choice for 'base' and 'novel' or 'novel_few'
        self.training_mode = 'novel_few'

        self.tag = format_tag
        self.pid_begin = pid_begin
        self.NWAY = NWAY
        self.KSHOT = KSHOT

        self.base_label2index, self.novel_label2index, self.selected_label2inds = buildLabelIndex(Way=self.NWAY, seed=1)

        if self.tag == 'tensor':
            self.dataset_dir = osp.join(root, self.root_folder, 'tensor', pretrained)
        else:  
            self.dataset_dir = osp.join(root, self.root_folder)

        self.rgb_dir = osp.join(self.dataset_dir, 'extended_photo')
        self.sketch_dir = osp.join(self.dataset_dir, 'sketch')

        self._check_before_run()
        train, val, query, gallery = self._process_dir(self.rgb_dir, self.sketch_dir, relabel=False, 
                                                  training_mode=self.training_mode, number_pthots=self.KSHOT, number_sketches=self.KSHOT, random_seed=0)
        
        print("=> Sketchy dataset loaded")
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
        if not osp.exists(self.rgb_dir):
            raise RuntimeError("'{}' is not available".format(self.rgb_dir))
        if not osp.exists(self.sketch_dir):
            raise RuntimeError("'{}' is not available".format(self.sketch_dir))

    def _process_dir(self, rgb_path, sketch_path, relabel=False, training_mode='base', number_pthots=1, number_sketches=1, eposido=100, number_way=5, random_seed=0):
        
        Way = number_way
        N = number_pthots
        M = number_sketches

        # Set the random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)

        # cross modal data, class folders
        photo_all_class = glob.glob(osp.join(rgb_path, "*"))
        sketch_all_class = glob.glob(osp.join(sketch_path, "*"))

        train_dataset = []
        val_dataset = []
        query_dataset = []
        gallery_dataset = []



        for photo_class_name in sorted(photo_all_class):
             # first get the class label
            photo_class = osp.basename(photo_class_name)  # like jellyfish
            photo_paths = glob.glob(osp.join(photo_class_name, "*.pt"))

            assert any(photo_class in sketch_cls for sketch_cls in sketch_all_class), "photo class {} not in sketch set".format(photo_class)

            sketch_class_name = osp.join(sketch_path, photo_class)
            draw_paths = glob.glob(osp.join(sketch_class_name, "*.pt"))


            if training_mode == 'base':
                # get the class label
                train_subset = train_classes_split

                if photo_class in train_subset:
                    
                    assert photo_class in self.base_label2index, "photo class {} not in label set".format(photo_class)
                    pid = self.base_label2index[photo_class]

                    # Shuffle the data to ensure random split
                    random.shuffle(photo_paths)

                    # Calculate the split indices
                    num_photos = len(photo_paths)

                    train_split_idx_photos = int(num_photos * 0.6)
                    val_split_idx_photos = int(num_photos * 0.8)

                    # Assign data to training, validation, and test sets
                    train_photo_paths = photo_paths[:train_split_idx_photos]
                    val_photo_paths = photo_paths[train_split_idx_photos:val_split_idx_photos]
                    test_photo_paths = photo_paths[val_split_idx_photos:]


                    # Add the training data
                    for photo_path in train_photo_paths:
                        train_dataset.append((photo_path, self.pid_begin + pid, 0, 'rgb'))

                    # for draw_path in train_sketch_paths:
                    #     train_dataset.append((draw_path, self.pid_begin + pid, 0, 'sketch'))

                    # Add the validation data
                    for photo_path in val_photo_paths:
                        val_dataset.append((photo_path, self.pid_begin + pid, 0, 'rgb'))
                    # for draw_path in val_sketch_paths:
                    #     val_dataset.append((draw_path, self.pid_begin + pid, 0, 'sketch'))

                    # Add the test data (gallery and query)
                    for photo_path in test_photo_paths:
                        gallery_dataset.append((photo_path, self.pid_begin + pid, 0, 'rgb'))
                    for draw_path in draw_paths:
                        query_dataset.append((draw_path, self.pid_begin + pid, 0, 'sketch'))
    
            elif training_mode == 'novel':
                # get the class label
                train_subset = novel_classes_split+val_classes_split
                if photo_class in train_subset:
                    assert photo_class in self.novel_label2index, "photo class {} not in label set".format(photo_class)
                    pid = self.novel_label2index[photo_class]
                    
                    # Select N random photos and M random sketches for training, rest for testing
                    if len(photo_paths) < N or len(draw_paths) < M:
                        raise ValueError(f"Not enough samples in class {photo_class} for few-shot training.")
                    
                    # Randomly select N photos and M sketches for training

                    # selected_photos = random.choices(photo_paths, k=N)  # Use random.choices for sampling with replacement
                    # selected_draws = random.choices(draw_paths, k=M)    # Use random.choices for sampling with replacement
                    # selected_photos = list(set(selected_photos))
                    # selected_draws = list(set(selected_draws))
                    
                    # or use random.sample for sampling without replacement 
                    selected_photos = random.sample(photo_paths, N)  # make the list unique
                    selected_draws = random.sample(draw_paths, M)

                    for photo_path in selected_photos:
                        train_dataset.append((photo_path, self.pid_begin + pid, 0, 'rgb'))
                    for draw_path in selected_draws:
                        train_dataset.append((draw_path, self.pid_begin + pid, 0, 'sketch'))
                    
                    # Remaining photos and sketches for testing
                    remaining_photos = [p for p in photo_paths if p not in selected_photos]
                    remaining_draws = [s for s in draw_paths if s not in selected_draws]

                    for photo_path in remaining_photos:
                        gallery_dataset.append((photo_path, self.pid_begin + pid, 0, 'rgb'))
                        val_dataset.append((photo_path, self.pid_begin + pid, 0, 'rgb'))
                    for draw_path in remaining_draws:
                        query_dataset.append((draw_path, self.pid_begin + pid, 0, 'sketch'))
                        val_dataset.append((draw_path, self.pid_begin + pid, 0, 'rgb'))
            
            elif training_mode == 'novel_few':
                if photo_class in self.selected_label2inds.keys():
                    pid = self.selected_label2inds[photo_class]
                    
                    # Select N random photos and M random sketches for training, rest for testing
                    if len(photo_paths) < N or len(draw_paths) < M:
                        raise ValueError(f"Not enough samples in class {photo_class} for few-shot training.")
                    
                    # Randomly select N photos and M sketches for training

                    # selected_photos = random.choices(photo_paths, k=N)  # Use random.choices for sampling with replacement
                    # selected_draws = random.choices(draw_paths, k=M)    # Use random.choices for sampling with replacement
                    # selected_photos = list(set(selected_photos))
                    # selected_draws = list(set(selected_draws))
                    
                    # or use random.sample for sampling without replacement 
                    selected_photos = random.sample(photo_paths, N)  # make the list unique
                    selected_draws = random.sample(draw_paths, M)

                    for photo_path in selected_photos:
                        train_dataset.append((photo_path, self.pid_begin + pid, 0, 'rgb'))
                    for draw_path in selected_draws:
                        train_dataset.append((draw_path, self.pid_begin + pid, 0, 'sketch'))
                    
                    # Remaining photos and sketches for testing
                    remaining_photos = [p for p in photo_paths if p not in selected_photos]
                    remaining_draws = [s for s in draw_paths if s not in selected_draws]

                    for photo_path in remaining_photos:
                        gallery_dataset.append((photo_path, self.pid_begin + pid, 0, 'rgb'))
                        val_dataset.append((photo_path, self.pid_begin + pid, 0, 'rgb'))
                    for draw_path in remaining_draws:
                        query_dataset.append((draw_path, self.pid_begin + pid, 0, 'sketch'))
                        val_dataset.append((draw_path, self.pid_begin + pid, 0, 'rgb'))

        return train_dataset, val_dataset, query_dataset, gallery_dataset
    

if __name__== '__main__':
    import sys
    sys.path.append('../')
    market_sketch = Sketchy(root="/home/stuyangz/Desktop/Zhengwei/github/datasets")