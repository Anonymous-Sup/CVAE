import time
import datetime
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.distributions.normal import Normal
from torch.distributions import MultivariateNormal
from tools.utils import AverageMeter
import torch.nn.functional as F
from utils import plot_histogram, plot_pair_seperate, plot_correlation_matrix, plot_scatter_1D, plot_scatter_2D
from utils import plot_histogram_seperate, print_gradients, plot_scatterNN, plot_epoch_Zdim
from tools.drawer import tSNE_plot

def train_cvae(run, config, model, classifier, criterion_cla, criterion_pair, criterion_recon,
              optimizer, trainloader, epoch, iteration_num):
    
    if config.DATA.TRAIN_FORMAT == 'novel':
        model.train()
        model.decoder.eval()
        classifier.train()
        drawer = tSNE_plot(num_query=None, trainplot=True)
        drawer.reset()
    elif config.DATA.TRAIN_FORMAT == 'novel_train_from_scratch':
        model.train()
        classifier.train()
        drawer = tSNE_plot(num_query=None, trainplot=True)
        drawer.reset()
    else:
        model.train()
        classifier.train()
    
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
        iteration_num += 1
        # convert fp16 tensor to fp32            
        if not config.TRAIN.AMP:
            imgs_tensor = imgs_tensor.float()

        imgs_tensor, pids, camids = imgs_tensor.cuda(), pids.cuda(), camids.cuda()

        run["train/batch/load_time"].append(time.time() - end)

        '''
        0422 norm or no norm for testing BatchNorm
        '''
        # imgs_tensor = model.norm(imgs_tensor)
        # recon_x, mean, log_var, z, x_pre, x_proj_norm, z_1, theta, logjacobin, domian_feature, flow_input= model.encode(imgs_tensor)   
        # x_pre, z, z_c, z_s, fusez_s, domian_feature, mean, log_var = model.encode(imgs_tensor)
        x_pre, mean, log_var, z_c, z_s, domian_feature, fusez_s, z, recon_x = model(imgs_tensor)
        
        if 'novel' in config.DATA.TRAIN_FORMAT:
            drawer.update((z_c, pids, clusterids))
            drawer.update_U(domian_feature)
        
        outputs = classifier(z_c)
        _, preds = torch.max(outputs.data, 1)
        cls_loss = criterion_cla(outputs, pids)

        pair_loss = criterion_pair(z_c, pids)

        base_dist = MultivariateNormal(torch.zeros_like(mean).cuda(), torch.eye(mean.size(1)).cuda())
        prior_p = base_dist.log_prob(z)
        prior = prior_p

        q_dist = Normal(mean, torch.exp(torch.clamp(log_var, min=-10) / 2))
        posterior_p = q_dist.log_prob(z)
        posterior = torch.sum(posterior_p, dim=-1)

        kl_loss = (posterior - prior).mean()          
        C = torch.clamp(torch.tensor(20.0) /
                            5000 * iteration_num, 0.0, 20.0)
        kl_loss = (kl_loss - C).abs()
        recon_loss = criterion_recon(recon_x, imgs_tensor)

        # regular_loss = criterion_regular(x_proj_norm)

        beta = 1.0
        gamma = 1.0
        if 'novel' in config.DATA.TRAIN_FORMAT:
            loss = recon_loss
            loss = loss + beta * kl_loss  # baseline no kl
            loss = loss + gamma * cls_loss
            # loss = loss + pair_loss

        elif config.MODEL.TRAIN_STAGE == 'klstage':
            loss = recon_loss  
            loss = loss + beta * kl_loss
            loss = loss + gamma * cls_loss
            # loss = loss + pair_loss

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
        
        z_collect = z if batch_idx == 0 else torch.cat((z_collect, z), dim=0)
        x_collect = x_pre if batch_idx == 0 else torch.cat((x_collect, x_pre), dim=0)
        zs_collect = fusez_s if batch_idx == 0 else torch.cat((zs_collect, fusez_s), dim=0)
        if (epoch+1) % 10 == 0 and batch_idx == len(trainloader)-1: 
            if 'reid' not in config.MODEL.TRAIN_STAGE:
                if 'novel' in config.DATA.TRAIN_FORMAT:
                    number_sample = 16
                    drawer.compute(run)
                else:
                    number_sample = 64
                
                plot_epoch_Zdim(run, z_collect, "0-Seperate dim of final cat z", number_sample)
                # plot_epoch_Zdim(run, z_collect, "0-Seperate dim of reparemeterized last z", last=True)
                plot_epoch_Zdim(run, x_collect, "0-Seperate dim of x_pre", number_sample)
                # plot_epoch_Zdim(run, x_collect, "0-Seperate dim of last x_pre", last=True)
                plot_epoch_Zdim(run, zs_collect, "0-Seperate dim of fusez_s", number_sample)
                
                plot_scatter_1D(run, prior_p, "1-prior_sample")
                plot_scatter_2D(run, posterior_p, "1-posterior_sample")

                plot_correlation_matrix(run, z, "1-correlation final cat z")
                plot_correlation_matrix(run, x_pre, "1-correlation x_pre")

                plot_histogram(run, mean, "2-mean")
                plot_histogram(run, log_var, "2-log_var")
                plot_histogram(run, z, "2-final cat z")
                plot_histogram(run, domian_feature, "3-domian_feature")
                plot_scatter_2D(run, domian_feature, "3-domian_feature scatter")
                plot_histogram(run, z_s, "4-Z_S")
                plot_histogram(run, z_c, "4-Z_C")
                plot_histogram(run, fusez_s, "4-fusez_s")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        batch_acc.update((torch.sum(preds == pids.data)).float()/pids.size(0), pids.size(0))
        batch_cls_loss.update(cls_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_kl_loss.update(kl_loss.item(), pids.size(0))
        batch_recon_loss.update(recon_loss.item(), pids.size(0))
        # batch_regular_loss.update(regular_loss.item(), pids.size(0))
        batch_loss.update(loss.item(), pids.size(0))
        batch_time.update(time.time() - end)

        run['train/batch/1_prior'].append(prior.mean().item())
        run['train/batch/1_posterior'].append(posterior.mean().item())
        run["train/batch/cls_loss"].append(cls_loss.item())
        run["train/batch/pair_loss"].append(pair_loss.item())
        run["train/batch/0_kl_loss"].append(kl_loss.item())
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
          'Pair Loss:{pair_loss.avg:.4f} '
          'KL Loss:{kl_loss.avg:.4f} '
          'Recon Loss:{bce_loss.avg:.4f} '
          'Acc:{acc.avg:.4f} '.format(
            epoch+1, batch_time=batch_time, data_time=data_time, 
            loss=batch_loss, cls_loss=batch_cls_loss,
            pair_loss=batch_pair_loss, kl_loss=batch_kl_loss,
            bce_loss=batch_recon_loss, acc=batch_acc)
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
    return iteration_num

