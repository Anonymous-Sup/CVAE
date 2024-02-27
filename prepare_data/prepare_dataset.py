import os
from shutil import copyfile
import argparse

######################################################################
# Prepare dataset for training
# You only need to change this line to your dataset download path
# --------------------------------------------------------------------

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_root', type=str, default='/home/zhengwei/github/datasets/')
argparser.add_argument('--dataset', type=str, default='DukeMTMC-reID', choices=['DukeMTMC-reID', 'Market-1501', 'MSMT17_V1', 'cuhk03', 'cuhk01'])

args = argparser.parse_args()

# download_path = 'D:\Datasets\DukeMTMC-reID'
download_path = os.path.join(args.data_root, args.dataset)

if 'cuhk' in download_path:
    suffix = 'png'
else:
    suffix = 'jpg'

if not os.path.isdir(download_path):
    print('please change the download_path')


save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)


def process_data(basepath, savepath):
    for root, dirs, files in os.walk(basepath, topdown=True):
        for name in files:
            if not name[-3:] == suffix:
                continue
            ID = name.split('_')
            src_path = basepath + '/' + name
            dst_path = savepath + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)
        print('dataset processed')

# For query 
query_path = download_path + '/query'
query_save_path = download_path + '/pytorch/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)
process_data(query_path, query_save_path)

# For gallery
gallery_path = download_path + '/bounding_box_test'
gallery_save_path = download_path + '/pytorch/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)
process_data(gallery_path, gallery_save_path)

# ---------------------------------------
# For train_all
train_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/pytorch/train_all'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
process_data(train_path, train_save_path)





