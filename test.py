import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from tools.eval_metrics import evaluate

@torch.no_grad()
def extract_midium_feature(model, dataloader, centroids):
    features, pids, camids, clothes_ids, centroids = [], torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids, batch_centroids) in enumerate(dataloader):
        
        # flip_imgs = torch.flip(imgs, [3])
        # imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        # batch_features = model(imgs)
        # batch_features_flip = model(flip_imgs)
        # batch_features += batch_features_flip
        # batch_features = F.normalize(batch_features, p=2, dim=1)

        pretrained_feautres = imgs
        pretrained_feautres = pretrained_feautres.cuda()
        recon_x, means, log_var, batch_features = model(pretrained_feautres, centroids[batch_centroids])
        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
        centroids = torch.cat((centroids, batch_centroids.cpu()), dim=0)
    features = torch.cat(features, 0)

    return features, pids, camids, clothes_ids, centroids


def test_cvae(run, config, model, queryloader, galleryloader, dataset):
    since = time.time()
    model.eval()
    # Extract features 
    qf, q_pids, q_camids, q_clothes_ids, q_centroids = extract_midium_feature(model, queryloader, dataset.query_centroids)
    gf, g_pids, g_camids, g_clothes_ids, g_q_centroids = extract_midium_feature(model, galleryloader, dataset.gallery_centroids)
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
    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    time_elapsed = time.time() - since
    print('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    since = time.time()
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Results ---------------------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    print('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    run["test/cmc"] = cmc
    run["test/mAP"] = mAP
    run["test/top1"] = cmc[0]
    run["test/top5"] = cmc[4]
    run["test/top10"] = cmc[9]

    return cmc[0]


