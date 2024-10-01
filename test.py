import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from tools.eval_metrics import evaluate
from data import build_singe_test_loader
from torch.cuda.amp import autocast
from tools.drawer import tSNE_plot
from utils import pair_plots, save_for_pairplot, idx2onehot
from tools.utils import AverageMeter
import json
import scipy.io
import os

@torch.no_grad()
def extract_midium_feature(batch_acc, reid_batch_acc, drawer, config, model, dataloader, classifier=None, classifier_reID=None, latent_z='z_c', final_epoch=False):
    
    features, feature_cls, pids, styleids, cls_result, all_imgs, all_recons, all_domains_y, all_img_paths, all_top10_scores, all_top10_labels = [], [], torch.tensor([]), torch.tensor([]), [], [], [], [], [], [], []
    
    if final_epoch:
        # Initialize dictionaries to store class accuracy and image paths with classification status
        class_acc_dict = {}  # Changed part
        class_img_paths = {}  # Changed part
    
    for batch_idx, (imgs, batch_pids, batch_styleids, batch_data_tags, batch_ima_path) in enumerate(dataloader):
        if not config.TRAIN.AMP:
            imgs = imgs.float()
        # flip_imgs = torch.flip(imgs, [3])
        # imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        # batch_features = model(imgs)
        # batch_features_flip = model(flip_imgs)
        # batch_features += batch_features_flip
        # batch_features = F.normalize(batch_features, p=2, dim=1)
        pretrained_features = imgs
        pretrained_features = pretrained_features.cuda()
        # recon_x, means, log_var, z, theta, logjcobin
        
        if config.DATA.DATASET == 'duke' or config.DATA.DATASET == 'msmt17':
            # expand 0 with the same shpe of batch_styleids
            used_styles = torch.zeros_like(batch_styleids)
        else:
            used_styles = batch_styleids

        used_styles = used_styles.cuda()
        # style_onehot = idx2onehot(used_styles, config.MODEL.STYLE_NUM)
        x_pre, mu, log_var, z_c, z_s, U, fusez_s, new_z, reconx = model(pretrained_features, used_styles)

        if latent_z == 'x_pre':
            retrieval_feature = x_pre
        elif latent_z == 'z_c':
            # only if using zc, do reid projection
            retrieval_feature = z_c
            # # for old testing
            # retrieval_feature = model.reid_projector(retrieval_feature)
            if config.DATA.TRAIN_FORMAT != 'novel_train_from_scratch':
                if config.MODEL.TRAIN_STAGE != 'klNocls_stage' and config.MODEL.TRAIN_STAGE != 'kl_cls_stage':
                    retrieval_feature = model.reid_projector(retrieval_feature)          
        elif latent_z == 'new_z':
            retrieval_feature = new_z
            # if config.DATA.TRAIN_FORMAT != 'novel_train_from_scratch':
            #     if config.MODEL.TRAIN_STAGE != 'klNocls_stage' and config.MODEL.TRAIN_STAGE != 'kl_cls_stage':
            #         retrieval_feature = model.reid_projector(retrieval_feature)
        elif latent_z == 'reconx':
            retrieval_feature = reconx
        elif latent_z == 'mu':
            retrieval_feature = mu

        if classifier != None:
            if config.DATA.TRAIN_FORMAT != 'novel_train_from_scratch':
                z_reid = model.reid_projector(z_c)
                z_c_proj = model.bottlenect(z_reid) 
                # z_c_proj = z_reid

                # z_c_proj = model.i2t_projector(z_c_reid)
                # z_c_proj = z_c
            else:
                z_c_proj = z_c
            outputs = classifier(z_c_proj)
            _, preds = torch.max(outputs.data, 1)
            pid_tensor = batch_pids.cuda()
            assert preds.shape == pid_tensor.shape
            batch_acc.update((torch.sum(preds == pid_tensor.data)).float()/pid_tensor.size(0), pid_tensor.size(0))
            
            # keep the top10 labels and scores for each class
            '''
            modifyed as 5
            '''
            batch_top10_scores, batch_top10_labels = torch.topk(outputs.data, 5)

            if classifier_reID != None:
                reid_feature = model.reid_projector(new_z)
                outputs_last = classifier_reID(reid_feature)
                _, preds_last = torch.max(outputs_last.data, 1)
                reid_batch_acc.update((torch.sum(preds_last == pid_tensor.data)).float()/pid_tensor.size(0), pid_tensor.size(0))

            if final_epoch:
                # Update class accuracy dictionary at the final epoch 
                for i in range(len(pid_tensor)):  
                    class_idx = pid_tensor[i].item()  
                    is_correct = preds[i] == pid_tensor[i] 
                    
                    if class_idx not in class_acc_dict:  
                        # class_acc_dict[class_idx] = {'correct': 0, 'total': 0}  
                        class_acc_dict[class_idx] = {'correct': 0, 'total': 0, 'top10_scores': [], 'top10_labels': []}
                    if class_idx not in class_img_paths:  
                        class_img_paths[class_idx] = [] 
                    
                    class_acc_dict[class_idx]['total'] += 1 
                    if is_correct:  
                        class_acc_dict[class_idx]['correct'] += 1  
                    
                    class_img_paths[class_idx].append((batch_ima_path[i], is_correct.item())) 
                    
                    # Get top 10 classification scores and corresponding labels
                    '''
                    modifyed as 5
                    '''
                    id_top10_scores, id_top10_labels = torch.topk(outputs.data[i], 5)
                    id_top10_labels = id_top10_labels.cpu().numpy()
                    id_top10_scores = id_top10_scores.cpu().numpy()
                    
                    class_acc_dict[class_idx]['top10_scores'].append(id_top10_scores)
                    class_acc_dict[class_idx]['top10_labels'].append(id_top10_labels)
        
        else:
            print("Ploting U&y, cls cant be None!")
            assert 1==0
            outputs = None

        # retrieval_feature = torch.cat((retrieval_feature, outputs), dim=-1)
        batach_features_norm = F.normalize(retrieval_feature, p=2, dim=1)
        
        features.append(batach_features_norm.cpu())
        # features.append(retrieval_feature.cpu())
        feature_cls.append(z_c_proj.cpu())


        all_top10_scores.append(batch_top10_scores.cpu())
        all_top10_labels.append(batch_top10_labels.cpu())

        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        styleids = torch.cat((styleids, batch_styleids.cpu()), dim=0)
        all_imgs.append(imgs.cpu())
        all_recons.append(reconx.cpu())

        cat_domain_y = torch.cat((U, outputs), dim=1)
        all_domains_y.append(cat_domain_y.cpu())
        all_img_paths += batch_ima_path

        drawer.update((batach_features_norm, batch_pids, batch_data_tags))
        drawer.update_U(U)
        
    all_top10_scores = torch.cat(all_top10_scores, 0)
    all_top10_labels = torch.cat(all_top10_labels, 0)
    features = torch.cat(features, 0)
    feature_cls = torch.cat(feature_cls, 0)
    all_imgs = torch.cat(all_imgs, 0)
    all_recons = torch.cat(all_recons, 0)
    all_domains_y = torch.cat(all_domains_y, 0)
    
    if final_epoch:
        class_accuracy = {}
        for class_idx, acc in class_acc_dict.items():
            accuracy = acc['correct'] / acc['total'] if acc['total'] > 0 else 0
            class_accuracy[class_idx] = {
                'accuracy': accuracy,
                'top10_scores': acc['top10_scores'],
                'top10_labels': acc['top10_labels']
            }
        del class_acc_dict
        # class_accuracy = {class_idx: acc['correct'] / acc['total'] for class_idx, acc in class_acc_dict.items()}
        return features, feature_cls, pids, styleids, all_imgs, all_recons, all_domains_y, all_img_paths, all_top10_scores, all_top10_labels, class_accuracy, class_img_paths

    # Assuming `classifier` is your model
    # for name, param in classifier.named_parameters():
    #     print(f'Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n')
    return features, feature_cls, pids, styleids, all_imgs, all_recons, all_domains_y, all_img_paths, all_top10_scores, all_top10_labels


"""
        with torch.no_grad():
            text_feature = text_feature_list[pids].float()
    
        nce_loss = criterion_nce(z_c, text_feature, pids, pids)
        logits = z_c @ text_feature.t()
        acc = (logits.max(1)[1] == pids).float().mean()
        acc_meter.update(acc, 1)
"""

def extract_midium_feature_withNCE(batch_acc, drawer, config, model, dataloader, classifier=None, text_embeddings=None, latent_z='z_c'):
    
    features, features_cat, pids, styleids, cls_result, all_imgs, all_recons = [], [], torch.tensor([]), torch.tensor([]), [], [], []
    
    for batch_idx, (imgs, batch_pids, batch_styleids, batch_data_tags, _) in enumerate(dataloader):
        if not config.TRAIN.AMP:
            imgs = imgs.float()

        pretrained_features = imgs
        pretrained_features = pretrained_features.cuda()
        # recon_x, means, log_var, z, theta, logjcobin
        
        if config.DATA.DATASET == 'duke' or config.DATA.DATASET == 'msmt17':
            # expand 0 with the same shpe of batch_styleids
            used_styles = torch.zeros_like(batch_styleids)
        else:
            used_styles = batch_styleids
        used_styles = used_styles.cuda()

        # style_onehot = idx2onehot(used_styles, config.MODEL.STYLE_NUM)
        x_pre, mu, log_var, z_c, z_s, U, fusez_s, new_z, reconx = model(pretrained_features, used_styles)
        
        reid_feature = model.reid_projector(z_c)
        z_c_proj = model.i2t_projector(reid_feature)

        if latent_z == 'x_pre':
            retrieval_feature = x_pre
        elif latent_z == 'z_c':
            retrieval_feature = z_c
            if config.MODEL.TRAIN_STAGE != 'klNocls_stage':
                retrieval_feature = model.reid_projector(retrieval_feature)
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

        retrieval_feature_cat = torch.cat((retrieval_feature, z_c_proj), dim=-1)

        batach_features_norm = F.normalize(retrieval_feature, p=2, dim=1)
        batch_catfeatures_norm = F.normalize(retrieval_feature_cat, p=2, dim=1)

        features.append(batach_features_norm.cpu())
        features_cat.append(batch_catfeatures_norm.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        styleids = torch.cat((styleids, batch_styleids.cpu()), dim=0)
        all_imgs.append(imgs.cpu())
        all_recons.append(reconx.cpu())

        drawer.update((batach_features_norm, batch_pids, batch_data_tags))
        drawer.update_U(U)
        

    features = torch.cat(features, 0)
    features_cat = torch.cat(features_cat, 0)
    all_imgs = torch.cat(all_imgs, 0)
    all_recons = torch.cat(all_recons, 0)
    # Assuming `classifier` is your model
    # for name, param in classifier.named_parameters():
    #     print(f'Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n')
    
    return features, features_cat, pids, styleids, all_imgs, all_recons

def convert_ndarray_to_list(data):
    """Recursively convert NumPy arrays to lists in a dictionary."""
    if isinstance(data, dict):
        return {k: convert_ndarray_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(v) for v in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

def convert_keys_to_string(input_dict):
    """
    Recursively converts dictionary keys to strings.
    """
    if isinstance(input_dict, dict):
        return {str(key): convert_keys_to_string(value) for key, value in input_dict.items()}
    elif isinstance(input_dict, list):
        return [convert_keys_to_string(element) for element in input_dict]
    elif isinstance(input_dict, tuple):
        return tuple(convert_keys_to_string(element) for element in input_dict)
    else:
        return input_dict

def evaluate_classification_accuracy(distmat, qf, gf, classifer, qids, gids):
    """
    Evaluate classification accuracy for person re-identification using the top-1 matched gallery.

    Parameters:
    - distmat: numpy.ndarray
      The distance matrix between query and gallery features.
    - qf: torch.Tensor
      The feature matrix for query images (size: [num_queries, feature_dim]).
    - gf: torch.Tensor
      The feature matrix for gallery images (size: [num_gallery, feature_dim]).
    - classifer: torch.nn.Module
      The classifier model to predict the class labels.

    Returns:
    - classification_accuracy: float
      The accuracy of the classification based on top-1 gallery matches.
    """
    m = qf.size(0)  # number of queries
    correct_classification_count = 0
    query_cls_count = 0
    gallery_cls_count = 0
    for i in range(m):
        # Find the index of the top-1 closest gallery image
        top1_index = np.argmin(distmat[i])

        # Extract the corresponding query and gallery features
        query_feature = qf[i].unsqueeze(0)  # (1, feature_dim)
        gallery_feature = gf[top1_index].unsqueeze(0)  # (1, feature_dim)

        # Pass both features through the classifier
        query_pred = classifer(query_feature)
        gallery_pred = classifer(gallery_feature)

        # Get the predicted labels
        _, query_label_pred = torch.max(query_pred, 1)
        _, gallery_label_pred = torch.max(gallery_pred, 1)

        # Compare the predicted labels
        if query_label_pred.item() == gallery_label_pred.item():
            correct_classification_count += 1
            # if query_label_pred.item() != qids[i]:
            #     print("Query prediction: {}, Gallery prediction: {}".format(query_label_pred.item(), gallery_label_pred.item()))
            #     print("Query label: {}, Gallery label: {}".format(qids[i], gids[top1_index]))

        if query_label_pred.item() == qids[i]:
            query_cls_count += 1

        if gallery_label_pred.item() == gids[top1_index]:
            gallery_cls_count += 1

    # Calculate the overall classification accuracy
    classification_accuracy = correct_classification_count / m
    query_cls_accuracy = query_cls_count / m
    gallery_cls_accuracy = gallery_cls_count / m
    return classification_accuracy, query_cls_accuracy, gallery_cls_accuracy

def test_cvae(run, config, model, queryloader, galleryloader, dataset, classifer=None, classifier_reID=None, text_embeddings=None, latent_z='fuse_z', final_epoch=False, cls_rerank=False):
    since = time.time()
    model.eval()
    drawer = tSNE_plot(len(dataset.query), trainplot=False)
    drawer.reset()
    if classifer != None:
        classifer.eval()
    if classifier_reID != None:
        classifier_reID.eval()
    # Extract features 
    print("==========Test with latent_z: {} =========".format(latent_z))
    q_batch_acc = AverageMeter()
    g_batch_acc = AverageMeter()
    q_reid_batch_acc = AverageMeter()
    g_reid_batch_acc = AverageMeter()
    if config.LOSS.USE_NCE:
        print("==========Test with NCE LOSS=========")
        qf, qf_cls, qf_cat, q_pids, q_camids, q_all_imgs, q_all_recons = extract_midium_feature_withNCE(q_batch_acc, drawer, config, model, queryloader, classifer, text_embeddings, latent_z)
        gf, gf_cls, gf_cat, g_pids, g_camids, g_all_imgs, g_all_recons = extract_midium_feature_withNCE(g_batch_acc, drawer, config, model, galleryloader, classifer, text_embeddings, latent_z)
    elif final_epoch:
        qf, qf_cls, q_pids, q_camids, q_all_imgs, q_all_recons, q_all_domains_y, q_all_img_path, q_10_scores, q_10_labels, q_class_acc_dict, q_class_path_dict = extract_midium_feature(q_batch_acc, q_reid_batch_acc, drawer, config, model, queryloader, classifer, classifier_reID, latent_z, final_epoch)
        gf, gf_cls, g_pids, g_camids, g_all_imgs, g_all_recons, g_all_domains_y, g_all_img_path, g_10_scores, g_10_labels, g_class_acc_dict, g_class_path_dict= extract_midium_feature(g_batch_acc, q_reid_batch_acc, drawer, config, model, galleryloader, classifer, classifier_reID, latent_z, final_epoch)
    else:
        qf, qf_cls, q_pids, q_camids, q_all_imgs, q_all_recons, q_all_domains_y, q_all_img_path, q_10_scores, q_10_labels = extract_midium_feature(q_batch_acc, q_reid_batch_acc, drawer, config, model, queryloader, classifer, classifier_reID, latent_z)
        gf, gf_cls, g_pids, g_camids, g_all_imgs, g_all_recons, g_all_domains_y, g_all_img_path, g_10_scores, g_10_labels = extract_midium_feature(g_batch_acc, g_reid_batch_acc, drawer, config, model, galleryloader, classifer, classifier_reID, latent_z)

    qf_norm = F.normalize(qf, p=2, dim=1)
    gf_norm = F.normalize(gf, p=2, dim=1)
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
    qf_cls, gf_cls = qf_cls.cuda(), gf_cls.cuda()
    qf_norm, gf_norm = qf_norm.cuda(), gf_norm.cuda()
    # Cosine similarity
    for i in range(m):
        # distmat[i] = (- torch.mm(qf[i:i+1], gf.t())).cpu()
        distmat[i] = (- torch.mm(qf_norm[i:i+1], gf_norm.t())).cpu()
    distmat = distmat.numpy()
    q_pids, q_camids = q_pids.numpy(), q_camids.numpy()
    g_pids, g_camids = g_pids.numpy(), g_camids.numpy()
    time_elapsed = time.time() - since
    print('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # former_merge_acc, former_q_pred, former_g_pred = evaluate_classification_accuracy(distmat, qf_cls, gf_cls, classifer, q_pids, g_pids)
    
    since = time.time()
    if config.DATA.DATASET == 'duke' or config.DATA.DATASET == 'msmt17':
        cmc, mAP, updatemat = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, q_all_img_path, g_all_img_path)
    else:
        if cls_rerank:
            if final_epoch:
                cmc, mAP, class_rank1_map_dict, all_results = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, q_all_img_path, g_all_img_path, nocam=True, final_epoch=final_epoch, cls_rerank=cls_rerank, q_10_scores=q_10_scores, q_10_labels=q_10_labels)
            else:
                cmc, mAP, updatemat = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, q_all_img_path, g_all_img_path, nocam=True, cls_rerank=cls_rerank, q_10_scores=q_10_scores, q_10_labels=q_10_labels)
        else:
            if final_epoch:
                cmc, mAP, class_rank1_map_dict, all_results = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, q_all_img_path, g_all_img_path, nocam=True, final_epoch=final_epoch)
            else:
                cmc, mAP, updatemat = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, q_all_img_path, g_all_img_path, nocam=True)

    
    # later_merge_acc, later_q_pred, later_g_pred = evaluate_classification_accuracy(updatemat, qf_cls, gf_cls, classifer, q_pids, g_pids)
    
    print("Results ---------------------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print("-----------------------------------------------------------")
    
    # print("Classification accuracy before cls-ranking: {:.1%}".format(former_merge_acc))
    # print("query accuracy after cls-ranking: {:.1%}".format(former_q_pred))
    # print("gallery accuracy after cls-ranking: {:.1%}".format(former_g_pred))

    # print("Classification accuracy after cls-ranking: {:.1%}".format(later_merge_acc))
    # print("query accuracy after cls-ranking: {:.1%}".format(later_q_pred))
    # print("gallery accuracy after cls-ranking: {:.1%}".format(later_g_pred))

    if config.LOSS.USE_NCE:
        m, n = qf_cat.size(0), gf_cat.size(0)
        distmat = torch.zeros((m,n))
        qf_cat, gf_cat = qf_cat.cuda(), gf_cat.cuda()
        # Cosine similarity
        for i in range(m):
            distmat[i] = (- torch.mm(qf_cat[i:i+1], gf_cat.t())).cpu()
        distmat = distmat.numpy()

        since = time.time()
        
        if config.DATA.DATASET == 'duke' or config.DATA.DATASET == 'msmt17':
            nocam = False
        else:
            nocam = True

        cmc_cat, mAP_cat = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, nocam=nocam)

        
        print("CAT Results ---------------------------------------------------")
        print('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc_cat[0], cmc_cat[4], cmc_cat[9], cmc_cat[19], mAP_cat))
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
    if classifier_reID != None:
        q_acc_reid = q_reid_batch_acc.avg
        g_acc_reid = g_reid_batch_acc.avg
        q_reid_batch_acc.merge(g_reid_batch_acc)
        q_g_acc_reid = q_reid_batch_acc.avg
        # total_acc = (q_acc + g_acc) / 2
        print("ReID Classifier results ---------------------------------------------------")
        print("Query acc: {:.1%} Gallery acc: {:.1%} Total acc: {:.1%}".format(q_acc_reid, g_acc_reid, q_g_acc_reid))
    if run != None:
        if final_epoch:
            mat_save_path = os.path.join(config.MODEL.RESUME, 'visual_results')
            if not os.path.exists(mat_save_path):
                os.makedirs(mat_save_path)

            # save q_class_acc_dict, q_class_path_dict and g_class_acc_dict, g_class_path_dict in a json file
            data_to_save = {
            "q_class_acc_dict": q_class_acc_dict,
            "q_class_path_dict": q_class_path_dict,
            "g_class_acc_dict": g_class_acc_dict,
            "g_class_path_dict": g_class_path_dict,
            'class_rank1_map_dict': class_rank1_map_dict
            }

            # Convert NumPy arrays to lists and keys to strings
            data_to_save = convert_ndarray_to_list(data_to_save)
            data_to_save = convert_keys_to_string(data_to_save)

            # Specify the filename
            filename = os.path.join(mat_save_path, "class_data.json")
            # Open the file and save the combined dictionary
            with open(filename, 'w') as f:
                json.dump(data_to_save, f, indent=4)

            
            # Save to Matlab for check
            gf_np, qf_np = gf.cpu().numpy(), qf.cpu().numpy()
            result = {'gallery_f':gf_np,'gallery_label':g_pids,'gallery_cam':g_camids, 'gallery_name': g_all_img_path ,'query_f':qf_np,'query_label':q_pids,'query_cam':q_camids, 'query_name': q_all_img_path}
            scipy.io.savemat(mat_save_path + '/pytorch_result.mat', result)
            
            # save all_results in a json file
            rank_results_path = os.path.join(mat_save_path, "rank_results.json")
            with open(rank_results_path, 'w') as f:
                json.dump(all_results, f, indent=4)
                
            del result
            del data_to_save            
        # else:
            # run["test/mAP"].append(mAP)
            # run["test/top1"].append(cmc[0])
            # run["test/top5"].append(cmc[4])
            # run["test/top10"].append(cmc[9])
            if config.DATA.DATASET == 'market1k':
                print("Jump TSNE in test")
                # drawer.compute(run)
    if latent_z == 'z_c':
        q_g_imgs = torch.cat((q_all_imgs, g_all_imgs), 0)
        q_g_recons = torch.cat((q_all_recons, g_all_recons), 0)
        q_g_features = torch.cat((qf, gf), 0)
        
        # pair_plots(config, q_g_imgs, q_g_features, "Q+G_X-Z_plots")
        # pair_plots(config, q_g_recons, q_g_features, "Q+G_Recons_Rx-Z_plots")

        # # save the q_g_imgs, q_g_recons, q_g_features, q_g_domains_y  in to a mat
        q_g_domains_y = torch.cat((q_all_domains_y, g_all_domains_y), 0)
        save_for_pairplot(len(q_all_imgs), q_g_imgs, q_g_recons, q_g_features, q_g_domains_y, config.MODEL.RESUME)
    return cmc, mAP, [q_acc, g_acc, q_g_acc]



def test_cvae_for_cls(run, config, model, valloader, queryloader, galleryloader, dataset, classifer=None, classifier_reID=None, text_embeddings=None, latent_z='fuse_z', final_epoch=False, cls_rerank=False):
    since = time.time()
    model.eval()

    drawer = tSNE_plot(len(dataset.query), trainplot=False)
    drawer.reset()

    if classifer != None:
        classifer.eval()
    if classifier_reID != None:
        classifier_reID.eval()

    # Extract features 
    print("==========Test with latent_z: {} =========".format(latent_z))
    q_batch_acc = AverageMeter()
    g_batch_acc = AverageMeter()
    q_reid_batch_acc = AverageMeter()
    g_reid_batch_acc = AverageMeter()

    val_batch_acc = AverageMeter()

    if config.LOSS.USE_NCE:
        print("==========Test with NCE LOSS=========")
        qf, qf_cls, qf_cat, q_pids, q_camids, q_all_imgs, q_all_recons = extract_midium_feature_withNCE(q_batch_acc, drawer, config, model, queryloader, classifer, text_embeddings, latent_z)
        gf, gf_cls, gf_cat, g_pids, g_camids, g_all_imgs, g_all_recons = extract_midium_feature_withNCE(g_batch_acc, drawer, config, model, galleryloader, classifer, text_embeddings, latent_z)
    
    elif final_epoch:
        qf, qf_cls, q_pids, q_camids, q_all_imgs, q_all_recons, q_all_domains_y, q_all_img_path, q_10_scores, q_10_labels, q_class_acc_dict, q_class_path_dict = extract_midium_feature(q_batch_acc, q_reid_batch_acc, drawer, config, model, queryloader, classifer, classifier_reID, latent_z, final_epoch)
        gf, gf_cls, g_pids, g_camids, g_all_imgs, g_all_recons, g_all_domains_y, g_all_img_path, g_10_scores, g_10_labels, g_class_acc_dict, g_class_path_dict= extract_midium_feature(g_batch_acc, q_reid_batch_acc, drawer, config, model, galleryloader, classifer, classifier_reID, latent_z, final_epoch)
    else:
        qf, qf_cls, q_pids, q_camids, q_all_imgs, q_all_recons, q_all_domains_y, q_all_img_path, q_10_scores, q_10_labels = extract_midium_feature(q_batch_acc, q_reid_batch_acc, drawer, config, model, queryloader, classifer, classifier_reID, latent_z)
        gf, gf_cls, g_pids, g_camids, g_all_imgs, g_all_recons, g_all_domains_y, g_all_img_path, g_10_scores, g_10_labels = extract_midium_feature(g_batch_acc, g_reid_batch_acc, drawer, config, model, galleryloader, classifer, classifier_reID, latent_z)
        vf, _, _, _, _, _, _, _, _, _ = extract_midium_feature(val_batch_acc, None, drawer, config, model, valloader, classifer, None, latent_z)
    
    torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))    
    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    print("Extracted features for validation set, obtained {} matrix".format(vf.shape))
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    time_elapsed = time.time() - since
    print('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if classifer != None:
        q_acc = q_batch_acc.avg
        g_acc = g_batch_acc.avg
        q_batch_acc.merge(g_batch_acc)
        q_g_acc = q_batch_acc.avg

        val_acc = val_batch_acc.avg
        # total_acc = (q_acc + g_acc) / 2
        print("Val Classifier results ---------------------------------------------------") 
        print("Total acc: {:.1%}".format(val_acc))

        print("Test Classifier results ---------------------------------------------------") 
        print("Query acc: {:.1%} Gallery acc: {:.1%} Total acc: {:.1%}".format(q_acc, g_acc, q_g_acc))

    # if classifier_reID != None:
    #     q_acc_reid = q_reid_batch_acc.avg
    #     g_acc_reid = g_reid_batch_acc.avg
    #     q_reid_batch_acc.merge(g_reid_batch_acc)
    #     q_g_acc_reid = q_reid_batch_acc.avg
    #     # total_acc = (q_acc + g_acc) / 2
    #     print("ReID Classifier results ---------------------------------------------------")
    #     print("Query acc: {:.1%} Gallery acc: {:.1%} Total acc: {:.1%}".format(q_acc_reid, g_acc_reid, q_g_acc_reid))
    # if run != None:
        # if final_epoch:
        #     mat_save_path = os.path.join(config.MODEL.RESUME, 'visual_results')
        #     if not os.path.exists(mat_save_path):
        #         os.makedirs(mat_save_path)

        #     # save q_class_acc_dict, q_class_path_dict and g_class_acc_dict, g_class_path_dict in a json file
        #     data_to_save = {
        #     "q_class_acc_dict": q_class_acc_dict,
        #     "q_class_path_dict": q_class_path_dict,
        #     "g_class_acc_dict": g_class_acc_dict,
        #     "g_class_path_dict": g_class_path_dict,
        #     }

        #     data_to_save = convert_keys_to_string(data_to_save)
        #     # Specify the filename
        #     filename = os.path.join(mat_save_path, "class_data.json")
        #     # Open the file and save the combined dictionary
        #     with open(filename, 'w') as f:
        #         json.dump(data_to_save, f, indent=4)

            
        #     # Save to Matlab for check
        #     gf, qf = gf.cpu().numpy(), qf.cpu().numpy()
        #     result = {'gallery_f':gf,'gallery_label':g_pids,'gallery_cam':g_camids, 'gallery_name': g_all_img_path ,'query_f':qf,'query_label':q_pids,'query_cam':q_camids, 'query_name': q_all_img_path}
        #     scipy.io.savemat(mat_save_path + '/pytorch_result.mat', result)
            
        #     # save all_results in a json file
        #     rank_results_path = os.path.join(mat_save_path, "rank_results.json")
        #     with open(rank_results_path, 'w') as f:
        #         json.dump(all_results, f, indent=4)
                
        #     del result
        #     del data_to_save            


    #     # else:
    #         # run["test/mAP"].append(mAP)
    #         # run["test/top1"].append(cmc[0])
    #         # run["test/top5"].append(cmc[4])
    #         # run["test/top10"].append(cmc[9])
    #         if config.DATA.DATASET == 'market1k':
    #             print("Jump TSNE in test")
    #             # drawer.compute(run)
    # if latent_z == 'z_c':
    #     q_g_imgs = torch.cat((q_all_imgs, g_all_imgs), 0)
    #     q_g_recons = torch.cat((q_all_recons, g_all_recons), 0)
    #     q_g_features = torch.cat((qf, gf), 0)
        
    #     pair_plots(config, q_g_imgs, q_g_features, "Q+G_X-Z_plots")
    #     pair_plots(config, q_g_recons, q_g_features, "Q+G_Recons_Rx-Z_plots")

    #     # # save the q_g_imgs, q_g_recons, q_g_features, q_g_domains_y  in to a mat
    #     # q_g_domains_y = torch.cat((q_all_domains_y, g_all_domains_y), 0)
    #     # save_for_pairplot(len(q_all_imgs), q_g_imgs, q_g_recons, q_g_features, q_g_domains_y, config.MODEL.RESUME)
    return val_acc, [q_acc, g_acc, q_g_acc]


@torch.no_grad()
def extract_test_feature_only(dataloader, final_epoch=False):
    features, pids, camids = [], torch.tensor([]), torch.tensor([])
    for batch_idx, (imgs, batch_pids, batch_camids, centroid,_) in enumerate(dataloader):
        features.append(imgs.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
    features = torch.cat(features, 0)
    print("Normalizing features")
    features = features / features.norm(dim=-1, keepdim=True)
    return features, pids, camids


def test_clip_feature(queryloader, galleryloader, dataset, final_epoch=False):
    since = time.time()
    # Extract features 
    if final_epoch:
        qf, q_pids, q_camids = extract_test_feature_only(queryloader)
        gf, g_pids, g_camids = extract_test_feature_only(galleryloader)
    else:
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
    if dataset == 'duke' or dataset == 'msmt17':
        cmc, mAP, _ = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    else:
        cmc, mAP, _ = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, nocam=True)
        
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
    parser.add_argument("--dataset", type=str, default="msmt17")
    parser.add_argument("--pretrained", type=str, default="AGWRes50", choices=["CLIPreid", "Transreid", "CLIPreidNew", 'AGWRes50'])

    args = parser.parse_args()

    query_loader, gallery_loader, dataset = build_singe_test_loader(args.data_root, args.dataset, pretrained=args.pretrained)
    test_clip_feature(query_loader, gallery_loader, args.dataset)

