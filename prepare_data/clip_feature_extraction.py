import torch
from PIL import Image
import open_clip
import os
import argparse
import neptune
import numpy as np
import tqdm

run = neptune.init_run(
    project="Zhengwei-Lab/NIPSTransferReID",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2ODIwNTQ4Yy0xZDA3LTRhNDctOTRmMy02ZjRlMmMzYmYwZjUifQ==",
)  # your credentials

# set only using GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_root', type=str, default='/home/zhengwei/github/datasets/')
argparser.add_argument('--dataset', type=str, default='DukeMTMC-reID', choices=['DukeMTMC-reID', 'Market-1501', 'MSMT17_V1', 'cuhk03', 'cuhk01'])
argparser.add_argument('--pre_model', type=str, default='CLIP-l2b')
args = argparser.parse_args()

if 'cuhk' in args.dataset:
    suffix = '.png'
else:
    suffix = '.jpg'


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
# tokenizer = open_clip.get_tokenizer('ViT-B-32')

model.eval()
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Context length:", context_length)
print("Vocab size:", vocab_size)

print(preprocess)
# run["model/preprocess"] = preprocess

def extract_features(model, preprocess, base_path, feature_path, suffix='.jpg'):
    # get all folders in the train_path
    dirs = os.listdir(base_path)
    # use tqdm to show the progress
    for dir in tqdm.tqdm(dirs):
        feature_root= os.path.join(feature_path, dir)
        os.makedirs(feature_root, exist_ok=True)
        
        root, _, files = next(os.walk(base_path + '/' + dir))
        for name in files:
            pre, ext = os.path.splitext(name)
            if ext != suffix:
                continue
            single_image_path = os.path.join(root, name)
            single_image = preprocess(Image.open(single_image_path)).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                single_image_features = model.encode_image(single_image.cuda())
                single_image_features /= single_image_features.norm(dim=-1, keepdim=True)
                torch.save(single_image_features, os.path.join(feature_root, pre + '.pt'))
        # break

print("==========Extracting feature form query ==========")
query_path = os.path.join(args.data_root, args.dataset, 'pytorch/query')
feature_path = os.path.join(args.data_root, args.dataset, 'tensor', args.pre_model, 'query')
run["data/query_path"] = query_path
extract_features(model, preprocess, query_path, feature_path, suffix)

print("==========Extracting feature form gallery ==========")
gallery_path = os.path.join(args.data_root, args.dataset, 'pytorch/gallery')
feature_path = os.path.join(args.data_root, args.dataset, 'tensor', args.pre_model, 'gallery')
run["data/gallery_path"] = gallery_path
extract_features(model, preprocess, gallery_path, feature_path, suffix)

print("==========Extracting feature form train_all ==========")
train_path = os.path.join(args.data_root, args.dataset, 'pytorch/train_all')
feature_path = os.path.join(args.data_root, args.dataset, 'tensor', args.pre_model, 'train_all' )
run["data/train_path"] = train_path
extract_features(model, preprocess, train_path, feature_path, suffix)

run.stop()