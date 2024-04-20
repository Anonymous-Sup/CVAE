import time
import datetime
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.distributions.normal import Normal
from torch.distributions import MultivariateNormal
from tools.utils import AverageMeter
import torch.nn.functional as F
from utils import plot_histogram, plot_pair_seperate, plot_correlation_matrix, plot_scatter_1D
from utils import plot_histogram_seperate, print_gradients, plot_scatterNN

def train_cvae(run, config, model, classifier, criterion_cla, criterion_pair, criterion_kl, criterion_recon, criterion_regular,
              optimizer, trainloader, epoch, centroids, early_stopping=None, scaler=None):
    
    if not config.TRAIN.AMP:
        centroids = centroids.float()
        print("=> centroids.dtype: {}".format(centroids.dtype))

    if config.DATA.TRAIN_FORMAT == 'novel':
        model.train()
        model.VAE.decoder.eval()
        classifier.train()
    else:
        if 'reid' in config.MODEL.TRAIN_STAGE:
            model.eval()
            classifier.train()
        else:
            model.train()
            classifier.eval()

    centroids.cuda()

    if config.MODEL.ONLY_CVAE_KL:
        print("=> Only CVAE KL")

    if config.MODEL.GAUSSIAN == 'MultivariateNormal':
        print("=> Use Multivariate Gaussian As the Prior")
        useMultiG = True
    elif config.MODEL.GAUSSIAN == 'Normal':
        print("=> Use Normal Gaussian As the Prior")
        useMultiG = False
    else:
        raise ValueError("Gaussian type {} is not supported".format(config.MODEL.GAUSSIAN))
    
    batch_cls_loss = AverageMeter()
    batch_cls_loss_theta = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_kl_loss = AverageMeter()
    batch_kld_theta = AverageMeter()
    batch_recon_loss = AverageMeter()
    batch_regular_loss = AverageMeter()
    batch_loss = AverageMeter()
    batch_acc = AverageMeter()
    batch_theta_acc = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    # run["train/epoch"].append(epoch)


    for batch_idx, (imgs_tensor, pids, camids, clusterids) in enumerate(trainloader):
        if epoch < 1 and batch_idx <= 20:
            plot_scatter_1D(run, clusterids, "5-domian index")
        # convert fp16 tensor to fp32
            
        if not config.TRAIN.AMP:
            imgs_tensor = imgs_tensor.float()

        imgs_tensor, pids, camids, clusterids = imgs_tensor.cuda(), pids.cuda(), camids.cuda(), clusterids.cuda()

        run["train/batch/load_time"].append(time.time() - end)

        imgs_tensor = model.norm(imgs_tensor)

        # model = model.to(imgs_tensor.dtype)
        # classifier = classifier.to(imgs_tensor.dtype)
        # return recon_x, means, log_var, z_0, z_1, theta, logjcobin
        
        """
        if amp is enabled, the forward pass will be autocast

        else not 
        """
        if config.TRAIN.AMP:
            with autocast():
                recon_x, mean, log_var, z, x_proj, z_1, theta, logjacobin, domian_feature, flow_input= model(imgs_tensor, centroids[clusterids])
            
                outputs = classifier(x_proj)
                outputs_theta = classifier(theta)

                _, preds = torch.max(outputs.data, 1)
                _, preds_theta = torch.max(outputs_theta.data, 1)

                cls_loss = criterion_cla(outputs, pids)
                cls_loss_theta = criterion_cla(outputs_theta, pids)

                pair_loss = criterion_pair(x_proj, pids)

                if config.MODEL.ONLY_CVAE_KL:
                    
                    posterior = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                    
                    base_dist = Normal(torch.zeros_like(mean), torch.ones_like(log_var))    
                    prior = torch.sum(base_dist.log_prob(theta + 1e-8), dim=-1)
                
                    kl_loss = (posterior-prior).mean()
                    kld_theta = kl_loss
                else:
                    base_dist = Normal(torch.zeros_like(mean), torch.ones_like(log_var))
                    prior = torch.sum(base_dist.log_prob(theta + 1e-8), dim=-1) + logjacobin.sum(-1)
        
                    # q0 = Normal(mean, torch.exp(0.5 * log_var))
                    
                    q0 = Normal(mean, torch.clamp(torch.exp(0.5 * log_var), min=1e-8))
                    posterior = torch.sum(q0.log_prob(z), dim=-1)
                    kl_loss = (posterior - prior).mean()

                    kl_loss = kl_loss.clamp(2.0)            
                    kld_theta = kl_loss
        
                recon_loss = criterion_recon(recon_x, imgs_tensor)

                beta = 0.05
                # loss = cls_loss  + beta *(kl_loss + kld_theta) + recon_loss

                loss = recon_loss + beta * kl_loss 
                    # loss = loss + cls_loss
        else:
            if config.MODEL.USE_CENTROID:
                domain_index = centroids[clusterids]
            else:
                domain_index = clusterids
            
            if 'reid' in config.MODEL.TRAIN_STAGE:
                with torch.no_grad():
                    recon_x, mean, log_var, z, x_proj, x_proj_norm, z_1, theta, logjacobin, domian_feature, flow_input= model(imgs_tensor, domain_index)
            else:
                recon_x, mean, log_var, z, x_proj, x_proj_norm, z_1, theta, logjacobin, domian_feature, flow_input= model(imgs_tensor, domain_index)
            

            recon_x = model.norm(recon_x)
            # (64, 702)
            outputs = classifier(x_proj_norm)
            outputs_theta = classifier(theta)

            _, preds = torch.max(outputs.data, 1)
            _, preds_theta = torch.max(outputs_theta.data, 1)

            cls_loss = criterion_cla(outputs, pids)
            cls_loss_theta = criterion_cla(outputs_theta, pids)

            pair_loss = criterion_pair(outputs, pids)

            if config.MODEL.ONLY_CVAE_KL:  
                posterior = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                
                if useMultiG:
                    # mean(64,12), 
                    base_dist = MultivariateNormal(torch.zeros_like(mean).cuda(), torch.eye(mean.size(1)).cuda())
                    # Here is a Jcobin! 
                    prior = base_dist.log_prob(theta)
                    # prior = prior + logjacobin.sum(-1)
                else:
                    base_dist = Normal(torch.zeros_like(mean), torch.ones_like(log_var))    
                    prior = torch.sum(base_dist.log_prob(theta), dim=-1)
                    # prior = prior + logjacobin.sum(-1)
            
                kl_loss = (posterior - prior).mean()
                kl_loss = kl_loss.clamp(2.0)
                kld_theta = kl_loss
            else:
                if useMultiG:
                    base_dist = MultivariateNormal(torch.zeros_like(mean).cuda(), torch.eye(mean.size(1)).cuda())
                    # (64)
                    prior = base_dist.log_prob(theta) + logjacobin.sum(-1)
                else:
                    base_dist = Normal(torch.zeros_like(mean), torch.ones_like(log_var))
                    # (64,12)
                    prior = torch.sum(base_dist.log_prob(theta), dim=-1) + logjacobin.sum(-1)
    
                # Normal for each z_i
                q0 = Normal(mean, torch.exp(0.5 * log_var))
                # if is independent, log(q) = logq_1 + logq_2 + ... + logq_n
                posterior = torch.sum(q0.log_prob(z), dim=-1)

                kl_loss = (posterior - prior).mean()
                kl_loss = kl_loss.clamp(2.0)            
                kld_theta = kl_loss

            recon_loss = criterion_recon(recon_x, imgs_tensor)

            # regular_loss = criterion_regular(x_proj_norm)


            beta = 0.01
            gamma = 0.5
            if config.DATA.TRAIN_FORMAT == 'novel':
                loss = recon_loss
                loss = loss + beta * kl_loss
                loss = loss + gamma * cls_loss
                loss = loss + pair_loss

            elif config.MODEL.TRAIN_STAGE == 'klstage':
                loss = recon_loss  
                loss = loss + beta * kl_loss    # for baseline there is only a reconsturction loss
                # loss = loss + regular_loss
                # loss = loss + pair_loss
                # loss = loss + cls_loss
            elif config.MODEL.TRAIN_STAGE == 'reidstage':
                loss = cls_loss
                # loss = pair_loss
                loss = 0.5 * loss + pair_loss
        
        # if early_stopping(kl_loss):
        #     print("Early stopping at epoch: {}".format(epoch))
        #     return False

        # # print every tensor for monitoring
        # print("mean: {}".format(mean[0, :10]))
        # print("log_var: {}".format(log_var[0, :10]))
        # print("z: {}".format(z[0, :10]))
        # # print("logjacobin: {}".format(logjacobin[0, :10]))
        # print("domian_feature: {}".format(domian_feature[0, :10]))
        # # print("flow_input: {}".format(flow_input[0, :10]))
        # # print("z_1: {}".format(z_1[0, :10]))
        # print("imgs_tensor: {}".format(imgs_tensor[0, :10]))
        # print("recon_x: {}".format(recon_x[0, :10]))
        
        # print("prior:{}".format(prior))
        # print("theta: {}".format(theta[0, :10]))
        # print("p_theta:{}".format(base_dist.log_prob(theta)))
        # print("logjacobin:{}".format(logjacobin))
        # print("posterior:{}".format(posterior))

        # if is the last batch
        if batch_idx == len(trainloader)-1:
            if 'reid' not in config.MODEL.TRAIN_STAGE:
                plot_scatterNN(run, x_proj, "0-N by N for z")
                plot_scatterNN(run, x_proj_norm, "0-N by N for norm_z")
                
                plot_correlation_matrix(run, x_proj_norm, "1-correlation z")
                # print("image_tensor: {}".format(imgs_tensor[0, :10]))
                # print("x_proj_norm.shape:{}, {}".format(x_proj_norm.shape, x_proj_norm[0, :]))
                plot_pair_seperate(run, x_proj_norm, "1-spedistribute z")

                plot_correlation_matrix(run, theta, "1-correlation theta")
                plot_pair_seperate(run, theta, "1-spedistribute theta")

                plot_histogram(run, mean, "2-mean")
                plot_histogram(run, log_var, "2-log_var")
                plot_histogram(run, z, "2-reparameterized z_0")
                plot_histogram(run, domian_feature, "3-domian_feature")
                print("domian_feature+Z: {}".format(flow_input[0, :2, -10:])) # print the last 10 elements
                plot_histogram(run, z_1, "3-flowinput-z_1")
                plot_histogram(run, theta, "4-theta")
                plot_histogram(run, logjacobin, "4-logjacobin")
                # plot_histogram(run, recon_x, "recon_x")
                # plot_histogram(run, imgs_tensor, "imgs_tensor")
                # plot_histogram(run, flow_input, "flow_input")

        optimizer.zero_grad()
        if config.TRAIN.AMP:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # print("Gradients before clipping:")
            # print_gradients(classifier)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        
        batch_acc.update((torch.sum(preds == pids.data)).float()/pids.size(0), pids.size(0))
        batch_theta_acc.update((torch.sum(preds_theta == pids.data)).float()/pids.size(0), pids.size(0))
        batch_cls_loss.update(cls_loss.item(), pids.size(0))
        batch_cls_loss_theta.update(cls_loss_theta.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_kl_loss.update(kl_loss.item(), pids.size(0))
        batch_kld_theta.update(kld_theta.item(), pids.size(0))
        batch_recon_loss.update(recon_loss.item(), pids.size(0))
        # batch_regular_loss.update(regular_loss.item(), pids.size(0))
        batch_loss.update(loss.item(), pids.size(0))
        batch_time.update(time.time() - end)

        run['train/batch/1_prior'].append(prior.mean().item())
        run['train/batch/1_posterior'].append(posterior.mean().item())
        run["train/batch/cls_loss"].append(cls_loss.item())
        run["train/batch/pair_loss"].append(pair_loss.item())
        run["train/batch/0_kl_loss"].append(kl_loss.item())
        run["train/batch/kld_theta"].append(kld_theta.item())
        run["train/batch/0_recon_loss"].append(recon_loss.item())
        # run["train/batch/0_regular_loss"].append(regular_loss.item())
        run["train/batch/loss"].append(loss.item())
        run["train/batch/acc"].append((torch.sum(preds == pids.data)).float()/pids.size(0))
        # run["train/batch/batch_time"].append(time.time() - end)
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
    run["train/epoch/kl_loss"].append(batch_kl_loss.avg)
    # run["train/epoch/kld_theta"].append(batch_kld_theta)
    run["train/epoch/recon_loss"].append(batch_recon_loss.avg)
    return True

