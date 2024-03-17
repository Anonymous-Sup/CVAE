import time
import datetime
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.distributions.normal import Normal
from tools.utils import AverageMeter
import torch.nn.functional as F
from utils import plot_histogram

def kl_divergence(z_0, z_1):
    # Calculate the mean and log-variance of z_0 along the batch dimension
    mean_z0 = torch.mean(z_0, dim=0)
    log_var_z0 = torch.log(torch.var(z_0, dim=0, unbiased=False) + 1e-5)
    
    # Calculate the mean and log-variance of z_1 along the batch dimension
    mean_z1 = torch.mean(z_1, dim=0)
    log_var_z1 = torch.log(torch.var(z_1, dim=0, unbiased=False) + 1e-5)
    
    # Calculate the KL divergence between z_1 and z_0 for each feature
    kl_div = 0.5 * torch.sum(log_var_z0 - log_var_z1 + (torch.exp(log_var_z1) + (mean_z1 - mean_z0)**2) / torch.exp(log_var_z0) - 1)
    
    return kl_div

def train_cvae(run, config, model, classifier, criterion_cla, criterion_pair, criterion_kl, criterion_recon, 
              optimizer, trainloader, epoch, centroids, scaler=None):
    
    model.train()
    classifier.train()
    centroids.cuda()

    only_cvae = False
    if only_cvae:
        print("=> Only CVAE and its loss")
    batch_cls_loss = AverageMeter()
    batch_cls_loss_theta = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_kl_loss = AverageMeter()
    batch_kld_theta = AverageMeter()
    batch_recon_loss = AverageMeter()
    batch_loss = AverageMeter()
    batch_acc = AverageMeter()
    batch_theta_acc = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    # run["train/epoch"].append(epoch)

    for batch_idx, (imgs_tensor, pids, camids, clusterids) in enumerate(trainloader):
        imgs_tensor, pids, camids, clusterids = imgs_tensor.cuda(), pids.cuda(), camids.cuda(), clusterids.cuda()

        run["train/batch/load_time"].append(time.time() - end)

        # model = model.to(imgs_tensor.dtype)
        # classifier = classifier.to(imgs_tensor.dtype)
        # return recon_x, means, log_var, z_0, z_1, theta, logjcobin
        with autocast():
            recon_x, mean, log_var, z, z_1, theta, logjacobin, domian_feature, flow_input= model(imgs_tensor, centroids[clusterids])
        
            outputs = classifier(z)
            outputs_theta = classifier(theta)

            _, preds = torch.max(outputs.data, 1)
            _, preds_theta = torch.max(outputs_theta.data, 1)

            cls_loss = criterion_cla(outputs, pids)
            cls_loss_theta = criterion_cla(outputs_theta, pids)

            pair_loss = criterion_pair(z, pids)

            # # initial kl with N(0,1)
            # # kl_loss = criterion_kl(mean, log_var)
            # normal_kl = torch.nn.KLDivLoss(reduction='batchmean')

            # # Q0 and prior
            # q0 = Normal(mean, torch.exp((0.5 * log_var)))
            # prior = Normal(torch.zeros_like(mean), torch.ones_like(log_var))
            
            # # prior_tensor = prior.sample((theta.size(0),))
            # # theta = F.log_softmax(theta, dim=1)
            # # prior_tensor = F.softmax(prior_tensor, dim=1)
            # # kl_loss_1 = normal_kl(theta, prior_tensor)
            # # logjacobin = logjacobin.unsqueeze(1)
            # # kl_2 =  F.log_softmax(theta+logjacobin, dim=1)
            # # z = F.softmax(z, dim=1)
            # # kl_loss_2 = normal_kl(kl_2, z)

            # kld_theta = (
            #     -torch.sum(prior.log_prob(theta), dim=-1)
            #     + torch.sum(q0.log_prob(z), dim=-1)
            #     - logjacobin.view(-1)
            # )
            # logjacobin = logjacobin.unsqueeze(-1)
            # kld_theta = 0.5 * (-logjacobin - 1 + torch.exp(logjacobin) + theta.pow(2))
            # kl_loss_z = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())


            # mu_q = theta.mean(0)
            # sigma_q = torch.exp(logjacobin).mean(0)
            # kld_theta = 0.5 * (-sigma_q.log() - 1 + sigma_q + mu_q.pow(2))
            
            # Ensure sigma_q is positive and non-zero
            # sigma_q = torch.clamp(sigma_q, min=1e-8)
            # kld_theta = 0.5 * (sigma_q.log() + mu_q.pow(2) / sigma_q - 1)

            # prior = Normal(torch.zeros_like(mean), torch.ones_like(log_var))
            # log_likelihood = prior.log_prob(theta) + logjacobin.unsqueeze(-1)
            # # kld_theta = -log_likelihood.sum()
            # kld_theta = -torch.logsumexp(log_likelihood, dim=0)

            if only_cvae:
                kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                kl_loss = kl_loss.mean()
                kld_theta = kl_loss
            else:
                base_dist = Normal(torch.zeros_like(mean), torch.ones_like(log_var))
                prior = torch.sum(base_dist.log_prob(theta), dim=-1) + logjacobin
                q0 = Normal(mean, torch.exp(0.5 * log_var))
                posterior = torch.sum(q0.log_prob(z_1), dim=-1)

                kl_loss = (posterior - prior).mean()
                kld_theta = kl_loss
            # kl loss between z (prior) and z_1 (post)
            # kl_z_z1 = kl_divergence(z.detach(), z_1)
            # kld_theta = kl_z_z1.mean()

            # print("posterior: {}, prior: {}".format(posterior.mean(), prior.mean()))
            # print("mean: {}, log_var: {}, z: {}".format(mean.mean(), log_var.mean(), z.mean()))

            # print("mean: {}, log_var: {}, theta: {}".format(mean, log_var, theta))
            # bce or mse
            recon_loss = criterion_recon(recon_x, imgs_tensor)

            beta = 0.5
            # loss = cls_loss  + beta *(kl_loss + kld_theta) + recon_loss

            loss = recon_loss + kl_loss 
            # loss = loss + cls_loss


            # if is the last batch
            if batch_idx == len(trainloader)-1:
                plot_histogram(run, mean, "mean")
                plot_histogram(run, log_var, "log_var")
                plot_histogram(run, z, "z")
                plot_histogram(run, z_1, "z_1")
                plot_histogram(run, theta, "theta")
                plot_histogram(run, logjacobin, "logjacobin")
                plot_histogram(run, recon_x, "recon_x")
                plot_histogram(run, imgs_tensor, "imgs_tensor")
                plot_histogram(run, domian_feature, "domian_feature")
                plot_histogram(run, flow_input, "flow_input")

        optimizer.zero_grad()
        if config.TRAIN.AMP:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        batch_acc.update((torch.sum(preds == pids.data)).float()/pids.size(0), pids.size(0))
        batch_theta_acc.update((torch.sum(preds_theta == pids.data)).float()/pids.size(0), pids.size(0))
        batch_cls_loss.update(cls_loss.item(), pids.size(0))
        batch_cls_loss_theta.update(cls_loss_theta.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_kl_loss.update(kl_loss.item(), pids.size(0))
        batch_kld_theta.update(kld_theta.item(), pids.size(0))
        batch_recon_loss.update(recon_loss.item(), pids.size(0))
        batch_loss.update(loss.item(), pids.size(0))
        batch_time.update(time.time() - end)

        run["train/batch/cls_loss"].append(cls_loss.item())
        run["train/batch/pair_loss"].append(pair_loss.item())
        run["train/batch/kl_loss"].append(kl_loss.item())
        run["train/batch/kld_theta"].append(kld_theta.item())
        run["train/batch/recon_loss"].append(recon_loss.item())
        run["train/batch/loss"].append(loss.item())
        run["train/batch/acc"].append((torch.sum(preds == pids.data)).float()/pids.size(0))
        run["train/batch/batch_time"].append(time.time() - end)
        end = time.time()

    print('Epoch:{0} '
          'Time:{batch_time.sum:.1f} '
          'Data:{data_time.sum:.1f} '
          'Loss:{loss.avg:.4f} '
          'Cls Loss:{cls_loss.avg:.4f} '
          'Cls Loss Theta:{cls_loss_theta.avg:.4f} '
          'Pair Loss:{pair_loss.avg:.4f} '
          'KL Loss:{kl_loss.avg:.4f} '
          'KLD Theta:{kld_theta.avg:.4f} '
          'Recon Loss:{bce_loss.avg:.4f} '
          'Acc:{acc.avg:.4f} '
          'Theta Acc:{theta_acc.avg:.4f} '.format(
            epoch+1, batch_time=batch_time, data_time=data_time, 
            loss=batch_loss, cls_loss=batch_cls_loss, cls_loss_theta=batch_cls_loss_theta,
            pair_loss=batch_pair_loss, kl_loss=batch_kl_loss, kld_theta=batch_kld_theta,
            bce_loss=batch_recon_loss, acc=batch_acc, theta_acc=batch_theta_acc)
          )
    # run["train/epoch/loss"].append(batch_loss)
    # run["train/epoch/acc"].append(batch_acc)
    # run["train/epoch/theta_acc"].append(batch_theta_acc)
    # run["train/epoch/cls_loss"].append(batch_cls_loss)
    # run["train/epoch/cls_loss_theta"].append(batch_cls_loss_theta)
    # run["train/epoch/pair_loss"].append(batch_pair_loss)
    # run["train/epoch/kl_loss"].append(batch_kl_loss)
    # run["train/epoch/kld_theta"].append(batch_kld_theta)
    # run["train/epoch/bce_loss"].append(batch_bce_loss)

