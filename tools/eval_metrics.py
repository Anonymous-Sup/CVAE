import numpy as np


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


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, q_paths=None, g_paths=None, nocam=False, final_epoch=False):
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