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

def train_cvae(run, config, model, classifier, criterion_cla, criterion_pair, criterion_kl, criterion_recon, criterion_regular,
              optimizer, trainloader, epoch, centroids, early_stopping=None, scaler=None, use_adapter=False):
    
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
            classifier.train()

    centroids.cuda()

    batch_cls_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_recon_loss = AverageMeter()
    batch_loss = AverageMeter()
    batch_acc = AverageMeter()
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

        '''
        0422 norm or no norm for testing BatchNorm
        '''
        # imgs_tensor = model.norm(imgs_tensor)

        if config.MODEL.USE_CENTROID:
            domain_index = centroids[clusterids]
        else:
            domain_index = clusterids
        
        # return recon_x, x_pre, x_proj_norm, domain_feature, fusion_UZ
        if 'reid' in config.MODEL.TRAIN_STAGE:
            with torch.no_grad():
                recon_x, x_pre, x_proj_norm, domian_feature, fusion_UZ= model(imgs_tensor, domain_index, norm=True)
        else:
            recon_x, x_pre, x_proj_norm, domian_feature, fusion_UZ= model(imgs_tensor, domain_index, norm=True)
        
        '''
        0422 norm or no norm for testing BatchNorm
        '''
        # recon_x = model.norm(recon_x)
    
        outputs = classifier(fusion_UZ)
            
        _, preds = torch.max(outputs.data, 1)

        cls_loss = criterion_cla(outputs, pids)

        pair_loss = criterion_pair(fusion_UZ, pids)

        recon_loss = criterion_recon(recon_x, imgs_tensor)

        beta = 0.01
        gamma = 0.5

        # this is a hyperparameter
        # adaptive_weight =  min(2.0 / (float(epoch)+1.0), 0.1)
        adaptive_weight =  0.0

        if config.DATA.TRAIN_FORMAT == 'novel':
            loss = adaptive_weight * recon_loss
            loss = loss + cls_loss
            loss = loss + pair_loss

        elif config.MODEL.TRAIN_STAGE == 'klstage':
            loss = recon_loss  
            # loss = loss + regular_loss
            loss = loss + pair_loss
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
        
        x_collect = x_pre if batch_idx == 0 else torch.cat((x_collect, x_pre), dim=0)
        fusion_collect = fusion_UZ if batch_idx == 0 else torch.cat((fusion_collect, fusion_UZ), dim=0)
        
        if batch_idx == len(trainloader)-1:
            if 'reid' not in config.MODEL.TRAIN_STAGE:
                
                if config.DATA.TRAIN_FORMAT == 'novel':
                    plot_epoch_Zdim(run, x_collect, "0-Seperate dim of x_pre", num_samples=16)
                    plot_epoch_Zdim(run, fusion_collect, "0-Seperate dim of z_fusion", num_samples=16)
                else:
                    plot_epoch_Zdim(run, x_collect, "0-Seperate dim of x_pre")
                    plot_epoch_Zdim(run, fusion_collect, "0-Seperate dim of z_fusion")

                plot_correlation_matrix(run, x_pre, "1-correlation x_pre")
                plot_correlation_matrix(run, fusion_collect, "1-correlation z_fusion")

                plot_histogram(run, domian_feature, "3-domian_feature")
                plot_scatter_2D(run, domian_feature, "3-domian_feature scatter")
                print("UZ fusion: {}".format(fusion_UZ[0, -10:])) # print the last 10 elements
                
                # plot_histogram(run, z_1, "3-flowinput-z_1")
                # plot_histogram(run, theta, "4-theta")
                # plot_histogram(run, logjacobin, "4-logjacobin")
                # # plot_histogram(run, recon_x, "recon_x")
                # # plot_histogram(run, imgs_tensor, "imgs_tensor")
                # # plot_histogram(run, flow_input, "flow_input")

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
        batch_cls_loss.update(cls_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_recon_loss.update(recon_loss.item(), pids.size(0))
        # batch_regular_loss.update(regular_loss.item(), pids.size(0))
        batch_loss.update(loss.item(), pids.size(0))
        batch_time.update(time.time() - end)


        run["train/batch/cls_loss"].append(cls_loss.item())
        run["train/batch/pair_loss"].append(pair_loss.item())
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
          'Recon Loss:{bce_loss.avg:.4f} '
          'Acc:{acc.avg:.4f} '.format(
            epoch+1, batch_time=batch_time, data_time=data_time, 
            loss=batch_loss, 
            cls_loss=batch_cls_loss,
            pair_loss=batch_pair_loss,
            bce_loss=batch_recon_loss, 
            acc=batch_acc)
          )

    # run["train/epoch/loss"].append(batch_loss)
    # run["train/epoch/acc"].append(batch_acc)
    # run["train/epoch/theta_acc"].append(batch_theta_acc)
    # run["train/epoch/cls_loss"].append(batch_cls_loss)
    # run["train/epoch/cls_loss_theta"].append(batch_cls_loss_theta)
    # run["train/epoch/pair_loss"].append(batch_pair_loss)
    # run["train/epoch/kl_loss"].append(batch_kl_loss.avg)
    # run["train/epoch/kld_theta"].append(batch_kld_theta)
    run["train/epoch/recon_loss"].append(batch_recon_loss.avg)
    return True

