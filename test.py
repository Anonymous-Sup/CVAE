import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from tools.eval_metrics import evaluate
from data import build_singe_test_loader
from torch.cuda.amp import autocast

@torch.no_grad()
def extract_midium_feature(config, model, dataloader, classifier=None, latent_z='fuse_z'):
    
    features, pids, camids = [], torch.tensor([]), torch.tensor([])
    for batch_idx, (imgs, batch_pids, batch_camids, batch_centroids) in enumerate(dataloader):
        if not config.TRAIN.AMP:
            imgs = imgs.float()

        # flip_imgs = torch.flip(imgs, [3])
        # imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        # batch_features = model(imgs)
        # batch_features_flip = model(flip_imgs)
        # batch_features += batch_features_flip
        # batch_features = F.normalize(batch_features, p=2, dim=1)

        pretrained_feautres = imgs
        pretrained_feautres = pretrained_feautres.cuda()
        # recon_x, means, log_var, z, theta, logjcobin
        

        x_pre, z_0, z_c, new_z, reconx = model(pretrained_feautres)
        
        if latent_z == 'z_0':
            retrieval_feature = z_0
        elif latent_z == 'x_pre':
            retrieval_feature = x_pre
        elif latent_z == 'z_c':
            retrieval_feature = z_c
        elif latent_z == 'new_z':
            retrieval_feature = new_z
        elif latent_z == 'reconx':
            retrieval_feature = reconx

        batach_features_norm = F.normalize(retrieval_feature, p=2, dim=1)

        features.append(batach_features_norm.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
    features = torch.cat(features, 0)
    return features, pids, camids


def test_cvae(run, config, model, queryloader, galleryloader, dataset, classifer=None, latent_z='fuse_z'):
    since = time.time()
    model.eval()
    classifer.eval()
    # Extract features 
    print("==========Test with latent_z: {} =========".format(latent_z))
    qf, q_pids, q_camids = extract_midium_feature(config, model, queryloader, classifer, latent_z)
    gf, g_pids, g_camids = extract_midium_feature(config, model, galleryloader, classifer, latent_z)
    # Gather samples from different GPUs
    # torch.cuda.empty_cache()
    # qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids], len(dataset.query))
    # gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))    
    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Compute distance matrix between query and gallery
    since = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat[i] = (- torch.mm(qf[i:i+1], gf.t())).cpu()
    distmat = distmat.numpy()
    q_pids, q_camids = q_pids.numpy(), q_camids.numpy()
    g_pids, g_camids = g_pids.numpy(), g_camids.numpy()
    time_elapsed = time.time() - since
    print('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    since = time.time()
    if 'reid' in config.MODEL.TRAIN_STAGE:
        print("Computing CMC and mAP through Classifier")
    else:
        print("Computing CMC and mAP")
    if config.DATA.DATASET == 'market1k':
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, nocam=True)
    else:
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Results ---------------------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    print('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if run != None:
        run["test/mAP"].append(mAP)
        run["test/top1"].append(cmc[0])
        run["test/top5"].append(cmc[4])
        run["test/top10"].append(cmc[9])

    return cmc, mAP


@torch.no_grad()
def extract_test_feature_only(dataloader):
    features, pids, camids = [], torch.tensor([]), torch.tensor([])
    for batch_idx, (imgs, batch_pids, batch_camids, centroid) in enumerate(dataloader):
        features.append(imgs.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
    features = torch.cat(features, 0)
    print("Normalizing features")
    features = features / features.norm(dim=-1, keepdim=True)
    return features, pids, camids


def test_clip_feature(queryloader, galleryloader, dataset):
    since = time.time()
    # Extract features 
    qf, q_pids, q_camids = extract_test_feature_only(queryloader)
    gf, g_pids, g_camids = extract_test_feature_only(galleryloader)
    # Gather samples from different GPUs
    # torch.cuda.empty_cache()
    # qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids], len(dataset.query))
    # gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))    
    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Compute distance matrix between query and gallery
    since = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat[i] = (- torch.mm(qf[i:i+1], gf.t())).cpu()
    distmat = distmat.numpy()
    q_pids, q_camids = q_pids.numpy(), q_camids.numpy()
    g_pids, g_camids = g_pids.numpy(), g_camids.numpy()
    time_elapsed = time.time() - since
    print('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    since = time.time()
    print("Computing CMC and mAP")
    if dataset == 'market1k':
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, nocam=True)
    else:
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Results ---------------------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    print('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return cmc[0]

if __name__=='__main__':
    import argparse
    import os
    # set cuda 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # /home/zhengwei/Desktop/Zhengwei/Projects/datasets/
    # /home/zhengwei/github/datasets

    parser = argparse.ArgumentParser(description="Test feature")
    parser.add_argument("--data_root", type=str, default="/home/zhengwei/Desktop/Zhengwei/Projects/datasets/")
    parser.add_argument("--dataset", type=str, default="duke")
    parser.add_argument("--pretrained", type=str, default="AGWRes50", choices=["CLIPreid", "Transreid", "CLIPreidNew", 'AGWRes50'])

    args = parser.parse_args()

    query_loader, gallery_loader, dataset = build_singe_test_loader(args.data_root, args.dataset, pretrained=args.pretrained)
    test_clip_feature(query_loader, gallery_loader, args.dataset)

