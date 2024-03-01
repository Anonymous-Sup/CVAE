import time
import datetime
import torch
# from apex import amp


def train_cvae(run, config, model, classifier, criterion_cla, criterion_pair, criterion_kl, criterion_bce, 
              optimizer, trainloader, epoch, centroids):
    
    model.train()
    classifier.train()
    centroids.cuda()

    end = time.time()

    for batch_idx, (imgs_tensor, pids, camids, clusterids) in enumerate(trainloader):
        imgs_tensor, pids, camids, clusterids = imgs_tensor.cuda(), pids.cuda(), camids.cuda(), clusterids.cuda()

        run["train/batch/load_time"] = time.time() - end

        recon_x, mean, log_var, z, theta, logjacobin= model(imgs_tensor, centroids[clusterids])

        outputs = classifier(z)
        _, preds = torch.max(outputs.data, 1)

        cls_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(z, pids)
        kl_loss = criterion_kl(mean, log_var)
        bce_loss = criterion_bce(recon_x, imgs_tensor)

        loss = cls_loss + pair_loss + kl_loss + bce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        run["train/batch/cls_loss"].append(cls_loss.item()/pids.size(0))
        run["train/batch/pair_loss"].append(pair_loss.item()/pids.size(0))
        run["train/batch/kl_loss"].append(kl_loss.item()/pids.size(0))
        run["train/batch/bce_loss"].append(bce_loss.item()/pids.size(0))
        run["train/batch/loss"].append(loss.item()/pids.size(0))
        run["train/batch/acc"].append((torch.sum(preds == pids.data)).float()/pids.size(0)/pids.size(0))
        run["train/batch/batch_time"] = time.time() - end
        end = time.time()
    
    run["train/epoch"] = epoch
