import numpy as np
import torch

def compute_ap_cmc(index, good_index, junk_index, g_img_paths=None, g_ids=None, result=None, final_epoch=False):
    ap = 0
    cmc = np.zeros(len(index)) 
    
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    # Handle case when ngood == 0
    if len(rows_good) == 0:
        return 0, cmc # Returns AP as 0.0 and unchanged CMC

    cmc[rows_good[0]:] = 1.0
    actual_ngood = len(rows_good)
    for i in range(actual_ngood):
        d_recall = 1.0 / ngood
        # print("rows_good", rows_good, "ngood", ngood)
        precision = (i+1)*1.0/(rows_good[i]+1)
        # if rows_good[i]!=0:
        #     old_precision = i*1.0/rows_good[i]
        # else:
        #     old_precision=1.0
        # ap = ap + d_recall*(old_precision + precision)/2
        ap = ap + d_recall*precision
    if final_epoch:
        result['top1'] = str(index[0])
        result['top5'] = [str(i) for i in index[:5]]
        result['top10'] = [str(i) for i in index[:10]]
        temp1 = []
        temp2 = []
        for j in range(10):
            temp1.append(str(g_img_paths[index[j]]))
            temp2.append(str(g_ids[index[j]]))
        result['rank_names'] = temp1
        result['rank_label'] = temp2
        min_rank = np.min(rows_good)
        if min_rank < 1:
            result['is_top1'] = 1
        else:
            result['is_top1'] = 0
        if min_rank < 5:
            result['is_top5'] = 1
        else:
            result['is_top5'] = 0
        if min_rank < 10:
            result['is_top10'] = 1
        else:
            result['is_top10'] = 0

        return ap, cmc, result
    return ap, cmc

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtracting np.max(x) for numerical stability
    return e_x / e_x.sum(axis=0)


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, q_paths=None, g_paths=None, nocam=False, final_epoch=False, cls_rerank=False, q_10_scores=None, q_10_labels=None):
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    if final_epoch:
        class_metrics = {}
        all_result = {} 

    # if q_10_labels is not None:
    #     for i in range(num_q):
    #         top_10_labels = q_10_labels[i]
            
    #         # Sort distances and get sorted indices
    #         sorted_indices = np.argsort(distmat[i])
            
    #         # Identify mismatched samples (ranked but not in top 10)
    #         mismatched_indices = []
    #         for rank in sorted_indices[:5]:  # Only consider top 10 ranked samples
    #             if g_pids[rank] not in top_10_labels:
    #                 mismatched_indices.append(rank)
            
    #         # Add weight to the mismatched samples
    #         distmat[i, mismatched_indices] += 0.1  # Add weight to mismatched samples
    

    # summarize the postive values in the distance matrix
    print("Positive values in distance matrix:", np.sum(distmat > 0))
    print("Negative values in distance matrix:", np.sum(distmat < 0))

    adjust_num = 0
    if q_10_labels is not None:
        for i in range(num_q):
            # Get the correct gallery samples for the query
            correct_indices = np.where(g_pids == q_pids[i])[0]
            
            # Sort distances and get sorted indices
            sorted_indices = np.argsort(distmat[i])
            
            # Check if any correct sample is in the top 10
            correct_in_top5 = any(idx in sorted_indices[:5] for idx in correct_indices)
            
            correct_in_top1 = any(idx in sorted_indices[:1] for idx in correct_indices)

            # # If no correct sample is in the top 10, add weight to all top 10 samples
            # if not correct_in_top5:
            #     correct_label_indices_in_q10 = np.where(np.isin(q_10_labels[i], g_pids[correct_indices]))[0]
            #     for label_index in correct_label_indices_in_q10:
            #         gallery_index = np.where(g_pids.astype(int) == int(q_10_labels[i][label_index]))[0]
            #         distmat[i, gallery_index] -= 0.2 
              
            # if not correct_in_top1:
            #     # show the acc scores
            #     # softmax the socres
            #     print("Top 10 acc softmax scores for non-correct:", torch.exp(q_10_scores[i][:10]) / torch.sum(torch.exp(q_10_scores[i][:10])))
            # else:
            #     print("Top 10 acc softmax scores for correct:", torch.exp(q_10_scores[i][:10]) / torch.sum(torch.exp(q_10_scores[i][:10])))

            # softmax the socre first
            q_10_scores[i] = torch.softmax(torch.tensor(q_10_scores[i]), dim=-1)

            for label in q_10_labels[i][:1]:
                # Find the gallery index that matches the label
                gallery_index = np.where(g_pids.astype(int) == int(label))[0]
                assert len(gallery_index) > 0, f"Label {label} not found in gallery"
                q_10_scores_np = q_10_scores[i][0].cpu().numpy()
                distmat[i, gallery_index] -= 1  # Adjust the distance for these specific gallery indices 
                
                adjust_num += 1
    
    print(f"Adjusted {adjust_num} distances")
    index = np.argsort(distmat, axis=1)  # Re-sort after adjusting distances
    
    for i in range(num_q):
        if final_epoch:
            result = {}
            query_pid = int(q_pids[i])
            if query_pid not in class_metrics: 
                class_metrics[query_pid] = {'num_r1': 0, 'AP': 0, 'count': 0} 
        
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        if nocam:
            good_index = query_index
        else:
            camera_index = np.argwhere(g_camids==q_camids[i])
            good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        
        if good_index.size == 0:
            num_no_gt += 1
            continue

        if nocam:
            junk_index = []
        # remove gallery samples that have the same pid and camid with query
        else:
            junk_index = np.intersect1d(query_index, camera_index)
        
        if final_epoch:
            ap_tmp, CMC_tmp, result = compute_ap_cmc(index[i], good_index, junk_index, g_paths, g_pids, result, final_epoch)
            result['id'] = i
            result['label'] = str(q_pids[i])
            result['img_name'] = str(q_paths[i])
            all_result[i] = result
        else:
            ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        
        if CMC_tmp[0]==1:
            num_r1 += 1
            if final_epoch:
                class_metrics[query_pid]['num_r1'] += 1 

        CMC = CMC + CMC_tmp
        AP += ap_tmp

        if final_epoch:
            class_metrics[query_pid]['AP'] += ap_tmp  # Changed part
            class_metrics[query_pid]['count'] += 1  # Changed part

    if num_no_gt > 0:
        print("{} query imgs do not have groundtruth.".format(num_no_gt))

    CMC = CMC / (num_q - num_no_gt)
    mAP = AP / (num_q - num_no_gt)


    if final_epoch:
        # Compute per-class rank@1 and mAP
        class_rank1_map = {}
        for class_idx, metrics in class_metrics.items():
            class_rank1_map[class_idx] = { 
            'rank1': metrics['num_r1'] / metrics['count'] if metrics['count'] > 0 else 0, 
            'mAP': metrics['AP'] / metrics['count'] if metrics['count'] > 0 else 0 
        } 
        del class_metrics

        return CMC, mAP, class_rank1_map, all_result
    return CMC, mAP, distmat




def evaluate_backup(distmat, q_pids, g_pids, q_camids, g_camids, q_paths=None, g_paths=None, nocam=False, final_epoch=False, q_10_scores=None, q_10_labels=None):
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    if final_epoch:
        class_metrics = {}
        all_result = {} 

    for i in range(num_q):
        if final_epoch:
            result = {}
            query_pid = int(q_pids[i])
            if query_pid not in class_metrics: 
                class_metrics[query_pid] = {'num_r1': 0, 'AP': 0, 'count': 0} 
        
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        if nocam:
            good_index = query_index
        else:
            camera_index = np.argwhere(g_camids==q_camids[i])
            good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        
        if good_index.size == 0:
            num_no_gt += 1
            continue

        if nocam:
            junk_index = []
        # remove gallery samples that have the same pid and camid with query
        else:
            junk_index = np.intersect1d(query_index, camera_index)
        
        if final_epoch:
            ap_tmp, CMC_tmp, result = compute_ap_cmc(index[i], good_index, junk_index, g_paths, g_pids, result, final_epoch)
            result['id'] = i
            result['label'] = str(q_pids[i])
            result['img_name'] = str(q_paths[i])
            all_result[i] = result
        else:
            ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        
        if CMC_tmp[0]==1:
            num_r1 += 1
            if final_epoch:
                class_metrics[query_pid]['num_r1'] += 1 

        CMC = CMC + CMC_tmp
        AP += ap_tmp

        if final_epoch:
            class_metrics[query_pid]['AP'] += ap_tmp  # Changed part
            class_metrics[query_pid]['count'] += 1  # Changed part

    if num_no_gt > 0:
        print("{} query imgs do not have groundtruth.".format(num_no_gt))

    CMC = CMC / (num_q - num_no_gt)
    mAP = AP / (num_q - num_no_gt)


    if final_epoch:
        # Compute per-class rank@1 and mAP
        class_rank1_map = {}
        for class_idx, metrics in class_metrics.items():
            class_rank1_map[class_idx] = { 
            'rank1': metrics['num_r1'] / metrics['count'] if metrics['count'] > 0 else 0, 
            'mAP': metrics['AP'] / metrics['count'] if metrics['count'] > 0 else 0 
        } 
        del class_metrics

        return CMC, mAP, class_rank1_map, all_result
    return CMC, mAP