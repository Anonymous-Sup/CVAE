import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from tools.eval_metrics import evaluate
from data import build_singe_test_loader
from torch.cuda.amp import autocast
from tools.drawer import tSNE_plot
from utils import pair_plots, save_for_pairplot
from tools.utils import AverageMeter

@torch.no_grad()
def extract_midium_feature(batch_acc, drawer, config, model, dataloader, classifier=None, latent_z='z_c'):
    
    features, pids, camids, cls_result, all_imgs, all_recons, all_domains_y = [], torch.tensor([]), torch.tensor([]), [], [], [], []
    for batch_idx, (imgs, batch_pids, batch_camids, batch_centroids, _) in enumerate(dataloader):
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
        
        x_pre, mu, log_var, z_c, z_s, U, fusez_s, new_z, reconx = model(pretrained_feautres)
        # x_pre, z_0, z_c, new_z, reconx, U, mu = model(pretrained_feautres)
    
        if latent_z == 'x_pre':
            retrieval_feature = x_pre
        elif latent_z == 'z_c':
            retrieval_feature = z_c
        elif latent_z == 'new_z':
            retrieval_feature = new_z
        elif latent_z == 'reconx':
            retrieval_feature = reconx
        elif latent_z == 'mu':
            retrieval_feature = mu

        if classifier != None:
            outputs = classifier(z_c)
            _, preds = torch.max(outputs.data, 1)
            pid_tensor = batch_pids.cuda()
            assert preds.shape == pid_tensor.shape
            batch_acc.update((torch.sum(preds == pid_tensor.data)).float()/pid_tensor.size(0), pid_tensor.size(0))
        else:
            print("Ploting U&y, cls cant be None!")
            assert 1==0
            outputs = None

        batach_features_norm = F.normalize(retrieval_feature, p=2, dim=1)
        
        features.append(batach_features_norm.cpu())
        # features.append(retrieval_feature.cpu())

        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        all_imgs.append(imgs.cpu())
        all_recons.append(reconx.cpu())

        cat_domain_y = torch.cat((U, outputs), dim=1)
        all_domains_y.append(cat_domain_y.cpu())

        drawer.update((batach_features_norm, batch_pids, batch_centroids))
        drawer.update_U(U)
        

    features = torch.cat(features, 0)
    all_imgs = torch.cat(all_imgs, 0)
    all_recons = torch.cat(all_recons, 0)
    all_domains_y = torch.cat(all_domains_y, 0)
    # Assuming `classifier` is your model
    # for name, param in classifier.named_parameters():
    #     print(f'Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n')
    return features, pids, camids, all_imgs, all_recons, all_domains_y


"""
        with torch.no_grad():
            text_feature = text_feature_list[pids].float()
    
        nce_loss = criterion_nce(z_c, text_feature, pids, pids)
        logits = z_c @ text_feature.t()
        acc = (logits.max(1)[1] == pids).float().mean()
        acc_meter.update(acc, 1)
"""

def extract_midium_feature_withNCE(batch_acc, drawer, config, model, dataloader, classifier=None, text_embeddings=None, latent_z='z_c'):
    
    features, pids, camids, cls_result, all_imgs, all_recons = [], torch.tensor([]), torch.tensor([]), [], [], []
    
    for batch_idx, (imgs, batch_pids, batch_camids, batch_centroids, _) in enumerate(dataloader):
        if not config.TRAIN.AMP:
            imgs = imgs.float()

        pretrained_feautres = imgs
        pretrained_feautres = pretrained_feautres.cuda()
        # recon_x, means, log_var, z, theta, logjcobin
        
        x_pre, mu, log_var, z_c, z_s, U, fusez_s, new_z, reconx = model(pretrained_feautres)
        # x_pre, z_0, z_c, new_z, reconx, U, mu = model(pretrained_feautres)
        
        z_c_proj = model.i2t_projection(z_c)

        if latent_z == 'x_pre':
            retrieval_feature = x_pre
        elif latent_z == 'z_c':
            retrieval_feature = z_c
        elif latent_z == 'new_z':
            retrieval_feature = new_z
        elif latent_z == 'reconx':
            retrieval_feature = reconx
        elif latent_z == 'mu':
            retrieval_feature = mu

        if text_embeddings != None:
            logits = z_c_proj @ text_embeddings.t()
            target = batch_pids.cuda()
            acc = (logits.max(1)[1] == target).float().mean()
            batch_acc.update(acc, 1)
        else:
            outputs = None

        batach_features_norm = F.normalize(retrieval_feature, p=2, dim=1)
        features.append(batach_features_norm.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        all_imgs.append(imgs.cpu())
        all_recons.append(reconx.cpu())

        drawer.update((batach_features_norm, batch_pids, batch_centroids))
        drawer.update_U(U)
        

    features = torch.cat(features, 0)
    all_imgs = torch.cat(all_imgs, 0)
    all_recons = torch.cat(all_recons, 0)
    # Assuming `classifier` is your model
    # for name, param in classifier.named_parameters():
    #     print(f'Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n')
    return features, pids, camids, all_imgs, all_recons


def test_cvae(run, config, model, queryloader, galleryloader, dataset, classifer=None, text_embeddings=None, latent_z='fuse_z'):
    since = time.time()
    model.eval()
    drawer = tSNE_plot(len(dataset.query), trainplot=False)
    drawer.reset()
    if classifer != None:
        classifer.eval()
    # Extract features 
    print("==========Test with latent_z: {} =========".format(latent_z))
    q_batch_acc = AverageMeter()
    g_batch_acc = AverageMeter()
    if config.LOSS.USE_NCE:
        print("==========Test with NCE LOSS=========")
        qf, q_pids, q_camids, q_all_imgs, q_all_recons = extract_midium_feature_withNCE(q_batch_acc, drawer, config, model, queryloader, classifer, text_embeddings, latent_z)
        gf, g_pids, g_camids, g_all_imgs, g_all_recons = extract_midium_feature_withNCE(g_batch_acc, drawer, config, model, galleryloader, classifer, text_embeddings, latent_z)
    else:
        qf, q_pids, q_camids, q_all_imgs, q_all_recons, q_all_domains_y = extract_midium_feature(q_batch_acc, drawer, config, model, queryloader, classifer, latent_z)
        gf, g_pids, g_camids, g_all_imgs, g_all_recons, g_all_domains_y = extract_midium_feature(g_batch_acc, drawer, config, model, galleryloader, classifer, latent_z)
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
    if config.DATA.DATASET == 'market1k':
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, nocam=True)
    else:
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Results ---------------------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    print('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if classifer != None:
        q_acc = q_batch_acc.avg
        g_acc = g_batch_acc.avg
        q_batch_acc.merge(g_batch_acc)
        q_g_acc = q_batch_acc.avg
        # total_acc = (q_acc + g_acc) / 2
        print("Classifier results ---------------------------------------------------") 
        print("Query acc: {:.1%} Gallery acc: {:.1%} Total acc: {:.1%}".format(q_acc, g_acc, q_g_acc))
        
    if run != None:
        if latent_z == 'new_z':
            q_g_imgs = torch.cat((q_all_imgs, g_all_imgs), 0)
            q_g_recons = torch.cat((q_all_recons, g_all_recons), 0)
            q_g_features = torch.cat((qf, gf), 0)
            
            pair_plots(run, q_g_imgs, q_g_features, "Q+G X-Z plots")
            pair_plots(run, q_g_recons, q_g_features, "Q+G Recons Rx-Z plots")

            # save the q_g_imgs, q_g_recons, q_g_features, q_g_domains_y  in to a mat
            # q_g_domains_y = torch.cat((q_all_domains_y, g_all_domains_y), 0)
            # save_for_pairplot(len(q_all_imgs), q_g_imgs, q_g_recons, q_g_features, q_g_domains_y, config.MODEL.RESUME)
        else:
            run["test/mAP"].append(mAP)
            run["test/top1"].append(cmc[0])
            run["test/top5"].append(cmc[4])
            run["test/top10"].append(cmc[9])
            if config.DATA.DATASET == 'market1k':
                drawer.compute(run)
    return cmc, mAP, [q_acc, g_acc, q_g_acc]


@torch.no_grad()
def extract_test_feature_only(dataloader):
    features, pids, camids = [], torch.tensor([]), torch.tensor([])
    for batch_idx, (imgs, batch_pids, batch_camids, centroid,_) in enumerate(dataloader):
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

