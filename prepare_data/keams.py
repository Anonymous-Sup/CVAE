from fast_pytorch_kmeans import KMeans
import torch
import argparse
import numpy as np
import os
import os.path as osp
import glob
import re
from collections import Counter

argparse = argparse.ArgumentParser()
argparse.add_argument('--type', type=str, default='train_all', choices=['train_all', 'query', 'gallery'])
argparse.add_argument('--n_clusters', type=int, default=25)
argparse.add_argument('--data_path', type=str, default='/home/zhengwei/Desktop/Zhengwei/Projects/datasets/DukeMTMC-reID')
argparse.add_argument('--sim_mode', type=str, default='euclidean', choices=['euclidean', 'cosine'])
argparse.add_argument('--init_method', type=str, default='random', choices=['kmeans++', 'random', 'gaussian'])
args = argparse.parse_args()

def clustering(args, tensors):
    labels = kmeans.fit_predict(tensors)
    centroids = kmeans.centroids
    counter = Counter(labels.cpu().numpy())
    sorted_counter = sorted(counter.items(), key=lambda x: x[0])  # sort by element
    return labels, centroids, sorted_counter

dir_path =  osp.join(args.data_path, 'tensor', args.type)
tensor_paths = glob.glob(osp.join(dir_path, '*/*.pt'))

# 将路径下所有的tensor保存成一个全部tensor，维度为（数量，维度）
all_tensors = []
for tensor_path in tensor_paths:
    tensor = torch.load(tensor_path)
    all_tensors.append(tensor)
all_tensors = torch.cat(all_tensors, dim=0)
all_tensors = all_tensors.cuda()

# 使用kmeans聚类
kmeans = KMeans(n_clusters=args.n_clusters, mode=args.sim_mode, init_method=args.init_method, verbose=1)
print("==========Clustering==========")
labels, centroids, sorted_counter = clustering(args, all_tensors)

while len(sorted_counter) != args.n_clusters:
    print("Only have {} clusters, Different from the predefined cluster {}".format(len(sorted_counter), args.n_clusters))
    print("==========Reclustering==========")
    labels, centroids, sorted_counter = clustering(args, all_tensors)
print("Clustering Done!")
print(sorted_counter)
print("==============================")

# save the dic of name to label
path2label = {}
for i, tensor_path in enumerate(tensor_paths):
    file_name = osp.basename(tensor_path)
    file_name = re.sub(r'\.pt$', '', file_name)
    label = labels[i].item()
    path2label[file_name] = label

# 保存labels和centroids
result={}
result['labels'] = labels # shape=(num_samples,)
result['path2label'] = path2label 
result['centroids'] = centroids # shape=(num_clusters, feature_dim)


save_path = "./Kmeans_result/DukeMTMC-reID"
if not osp.exists(save_path):
    os.makedirs(save_path)
same_name = '{}_{}k_{}.pt'.format(args.type, args.n_clusters, args.sim_mode)
print('Keams results save to', osp.join(save_path, same_name))
torch.save(result, save_path)
print('Saving Success!')
