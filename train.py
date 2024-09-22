import time
import datetime
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.distributions.normal import Normal
from torch.distributions import MultivariateNormal
from tools.utils import AverageMeter
import torch.nn.functional as F
from utils import plot_histogram, plot_pair_seperate, plot_correlation_matrix, plot_scatter_1D, plot_scatter_2D
from utils import plot_histogram_seperate, print_gradients, plot_scatterNN, plot_epoch_Zdim, idx2onehot
from tools.drawer import tSNE_plot


def train_cvae(run, config, model, classifier, classifer_reid, criterion_cla, criterion_pair, criterion_recon, criterion_center,
              optimizer, optimizer_center, trainloader, epoch, iteration_num):
    
    drawer = tSNE_plot(num_query=None, trainplot=True)
    drawer.reset()
    if config.DATA.TRAIN_FORMAT == 'novel':
        if 'kl' not in config.MODEL.TRAIN_STAGE:
            model.eval()
            classifier.eval()
            model.bottlenect.train()
            if config.MODEL.TRAIN_STAGE == 'reid+cls_stage':
                model.reid_projector.train()
                model.i2t_projector.train()
                classifier.train()
            elif config.MODEL.TRAIN_STAGE == 'reidstage':
                model.reid_projector.train()
            elif config.MODEL.TRAIN_STAGE == 'CLSstage':
                model.i2t_projector.train()
                classifier.train()
        else:
            model.train()
            classifier.train()
            model.decoder.eval()
            model.bottlenect.eval()
            if config.MODEL.TRAIN_STAGE == 'kl_reid_stage':
                model.i2t_projector.eval()
                classifier.eval()
            elif config.MODEL.TRAIN_STAGE == 'kl_cls_stage':
                model.reid_projector.eval()
                model.bottlenect.train()
            elif config.MODEL.TRAIN_STAGE == 'klNocls_stage':
                model.i2t_projector.eval()
                model.reid_projector.eval()
                classifier.eval()
            
    elif config.DATA.TRAIN_FORMAT == 'novel_train_from_scratch':
        """
        Todo:
        This part aslo need to be seperate with two stages
        """
        model.train()
        classifier.train()
    else: # for base data
        if 'kl' in config.MODEL.TRAIN_STAGE:
            model.train()
            classifier.train()
            model.bottlenect.eval()
            if config.MODEL.TRAIN_STAGE == 'kl_reid_stage':
                model.i2t_projector.eval()
                classifier.eval()
            elif config.MODEL.TRAIN_STAGE == 'kl_cls_stage':
                model.reid_projector.eval()
                model.bottlenect.train()
            elif config.MODEL.TRAIN_STAGE == 'klNocls_stage':
                model.i2t_projector.eval()
                model.reid_projector.eval()
                classifier.eval()
        else:
            model.eval()
            classifier.eval()
            model.bottlenect.train()
            if config.MODEL.TRAIN_STAGE == 'reid+cls_stage':
                model.reid_projector.train()
                model.i2t_projector.train()
                classifier.train()
            if config.MODEL.TRAIN_STAGE == 'reidstage':
                model.reid_projector.train()
            elif config.MODEL.TRAIN_STAGE == 'CLSstage':
                model.i2t_projector.train()
                classifier.train()
                model.bottlenect.eval()

    
    batch_cls_loss = AverageMeter()
    # batch_cls_reid_loss = AverageMeter()
    batch_center_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_kl_loss = AverageMeter()
    # batch_kld_theta = AverageMeter()
    batch_recon_loss = AverageMeter()
    batch_loss = AverageMeter()
    batch_acc = AverageMeter()
    # batch_reid_acc = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    # run["train/epoch"].append(epoch)

    for batch_idx, (imgs_tensor, pids, style_ids, data_tag, _) in enumerate(trainloader):
        iteration_num += 1
        # convert fp16 tensor to fp32            
        if not config.TRAIN.AMP:
            imgs_tensor = imgs_tensor.float()

        imgs_tensor, pids, style_ids = imgs_tensor.cuda(), pids.cuda(), style_ids.cuda()

        '''
        0422 norm or no norm for testing BatchNorm
        '''
        # imgs_tensor = model.norm(imgs_tensor)
        # recon_x, mean, log_var, z, x_pre, x_proj_norm, z_1, theta, logjacobin, domian_feature, flow_input= model.encode(imgs_tensor)   
        # x_pre, z, z_c, z_s, fusez_s, domian_feature, mean, log_var = model.encode(imgs_tensor)

        if config.DATA.DATASET == 'duke' or config.DATA.DATASET == 'msmt17':
            style_ids = torch.zeros_like(style_ids)
            style_ids = style_ids.cuda()

        # styles_onehot =  idx2onehot(style_ids, config.MODEL.STYLE_NUM)
        x_pre, mean, log_var, z_c, z_s, domian_feature, fusez_s, z, recon_x = model(imgs_tensor, style_ids)

        if 'novel' in config.DATA.TRAIN_FORMAT and 'kl' in config.MODEL.TRAIN_STAGE:
            drawer.update((z_c, pids, data_tag))
            drawer.update_U(domian_feature)

        """
        Here should be noticed using which z for reid
        optional: z, z_c
        z for final, z_c for only content
        """
        z_reid = model.reid_projector(z_c)
        pair_loss = criterion_pair(z_reid, pids)
        center_loss = criterion_center(z_reid, pids)

        """
        Here should be noticed using which z for cls
        optional: z_c, z_reid
        z_c for direct, z_reid for linear structure
        """
        # z_c_proj = model.i2t_projector(z_reid)

        z_reid_bn = model.bottlenect(z_reid)
        # z_reid_bn = z_reid
        outputs = classifier(z_reid_bn)
        
        _, preds = torch.max(outputs.data, 1)
        cls_loss = criterion_cla(outputs, pids)

        if classifer_reid is not None:
            outputs_reid = classifer_reid(z_reid)
            _, preds_reid = torch.max(outputs_reid.data, 1)
            cls_loss_reid = criterion_cla(outputs_reid, pids)

        base_dist = MultivariateNormal(torch.zeros_like(mean).cuda(), torch.eye(mean.size(1)).cuda())
        prior_p = base_dist.log_prob(z)
        prior = prior_p

        q_dist = Normal(mean, torch.exp(torch.clamp(log_var, min=-10) / 2))
        posterior_p = q_dist.log_prob(z)
        posterior = torch.sum(posterior_p, dim=-1)

        kl_loss = (posterior - prior).mean()   
        if config.FEWSHOT.ENABLE and config.DATA.TRAIN_FORMAT == 'novel':
            C = torch.clamp(torch.tensor(50.0) /
                            5000 * iteration_num, 0.0, 20.0)   
        else:    
            C = torch.clamp(torch.tensor(20.0) /
                                5000 * iteration_num, 0.0, 20.0)
        kl_loss = (kl_loss - C).abs()
        recon_loss = criterion_recon(recon_x, imgs_tensor)

        # regular_loss = criterion_regular(x_proj_norm)

        beta = 1.0
        gamma = 1.0

        loss = 0.0
        if 'kl' in config.MODEL.TRAIN_STAGE:
            loss = recon_loss
            loss = loss + beta * kl_loss
            if config.MODEL.TRAIN_STAGE == 'kl_reid_stage':
                loss += pair_loss + center_loss
            elif config.MODEL.TRAIN_STAGE == 'kl_cls_stage':
                loss += cls_loss
            elif config.MODEL.TRAIN_STAGE == 'klNocls_stage':
                loss = loss
            else: # for all loss joint training 
                loss += cls_loss
                loss += pair_loss
                # loss += center_loss
        elif config.MODEL.TRAIN_STAGE == 'reid+cls_stage':
            loss += cls_loss
            loss += pair_loss
            loss += center_loss
        elif config.MODEL.TRAIN_STAGE == 'reidstage':
            loss = pair_loss 
            loss += center_loss
        elif config.MODEL.TRAIN_STAGE == 'CLSstage':
            loss = cls_loss


        z_collect = z if batch_idx == 0 else torch.cat((z_collect, z), dim=0)
        x_collect = x_pre if batch_idx == 0 else torch.cat((x_collect, x_pre), dim=0)
        zs_collect = fusez_s if batch_idx == 0 else torch.cat((zs_collect, fusez_s), dim=0)
        # if (epoch+1) % 10 == 0 and batch_idx == len(trainloader)-1:   
        # only for the last epoch
        if epoch+1 == config.TRAIN.MAX_EPOCH and batch_idx == len(trainloader)-1:
            if 'reid' not in config.MODEL.TRAIN_STAGE:
                if 'kl' in config.MODEL.TRAIN_STAGE:
                    number_sample = 16
                else:
                    number_sample = 64
                # (batch* samplebatch, dim)
                plot_epoch_Zdim(config, z_collect, "0_Seperate_dim_of_final_cat_z", number_sample)
                # plot_epoch_Zdim(config, z_collect, "0-Seperate dim of reparemeterized last z", last=True)
                plot_epoch_Zdim(config, x_collect, "0_Seperate_dim_of_x_pre", number_sample)
                # plot_epoch_Zdim(config, x_collect, "0-Seperate dim of last x_pre", last=True)
                plot_epoch_Zdim(config, zs_collect, "0_Seperate_dim_of_fusez_s", number_sample)
                
                plot_scatter_1D(config, prior_p, "1_prior_sample")
                plot_scatter_2D(config, posterior_p, "1_posterior_sample")

                plot_correlation_matrix(config, z, "1_correlation_final_cat_z")
                plot_correlation_matrix(config, x_pre, "1_correlation_x_pre")

                plot_histogram(config, mean, "2_mean")
                plot_histogram(config, log_var, "2_log_var")
                plot_histogram(config, z, "2_final_cat_z")
                plot_histogram(config, domian_feature, "3_domian_feature")
                plot_scatter_2D(config, domian_feature, "3_domian_feature_scatter")
                plot_histogram(config, z_s, "4_Z_S")
                plot_histogram(config, z_c, "4_Z_C")
                plot_histogram(config, fusez_s, "4_fusez_s")

        optimizer.zero_grad()
        if optimizer_center is not None:
            optimizer_center.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        if optimizer_center is not None:
            optimizer_center.step()

 
        batch_acc.update((torch.sum(preds == pids.data)).float()/pids.size(0), pids.size(0))
        # batch_reid_acc.update((torch.sum(preds_reid == pids.data)).float()/pids.size(0), pids.size(0))
        batch_loss.update(loss.item(), pids.size(0))
        batch_cls_loss.update(cls_loss.item(), pids.size(0))
        # print("all loss: {}".format(loss.item()))
        # print("cls_loss: {}".format(cls_loss.item()))
        # batch_cls_reid_loss.update(cls_loss_reid.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_center_loss.update(center_loss.item(), pids.size(0))
        batch_kl_loss.update(kl_loss.item(), pids.size(0))
        batch_recon_loss.update(recon_loss.item(), pids.size(0))
        # batch_regular_loss.update(regular_loss.item(), pids.size(0))
        
        batch_time.update(time.time() - end)
        

        # run['train/batch/1_prior'].append(prior.mean().item())
        # run['train/batch/1_posterior'].append(posterior.mean().item())
        # run["train/batch/cls_loss"].append(cls_loss.item())
        # run["train/batch/cls_loss_reid"].append(cls_loss_reid.item())
        # run["train/batch/pair_loss"].append(pair_loss.item())
        # run["train/batch/center_loss"].append(center_loss.item())
        # run["train/batch/0_kl_loss"].append(kl_loss.item())
        # run["train/batch/0_recon_loss"].append(recon_loss.item())
        # # run["train/batch/0_regular_loss"].append(regular_loss.item())
        # run["train/batch/loss"].append(loss.item())
        # run["train/batch/acc"].append((torch.sum(preds == pids.data)).float()/pids.size(0))
        # run["train/batch/reid_acc"].append((torch.sum(preds_reid == pids.data)).float()/pids.size(0))
        # run["train/batch/batch_time"].append(time.time() - end)
        end = time.time()

    print('Epoch:{0} '
          'Time:{batch_time.sum:.1f} '
          'Data:{data_time.sum:.1f} '
          'Loss:{loss.avg:.4f} '
          'Cls Loss:{cls_loss.avg:.4f} '
          'Pair Loss:{pair_loss.avg:.4f} '
          'Center Loss:{center_loss.avg:.4f} '
          'KL Loss:{kl_loss.avg:.4f} '
          'Recon Loss:{bce_loss.avg:.4f} '
          'Acc:{acc.avg:.4f} '.format(
            epoch+1, batch_time=batch_time, data_time=data_time, 
            loss=batch_loss, cls_loss=batch_cls_loss, 
            pair_loss=batch_pair_loss, center_loss=batch_center_loss, 
            kl_loss=batch_kl_loss, bce_loss=batch_recon_loss, acc=batch_acc)
          )
    if 'reid' not in config.MODEL.TRAIN_STAGE:
        if 'kl' in config.MODEL.TRAIN_STAGE:
            if (epoch+1) % 10 == 0:
                print("Jump TSNE")
                # drawer.compute(run)
    # run["train/epoch/loss"].append(batch_loss)
    # run["train/epoch/acc"].append(batch_acc)
    # run["train/epoch/theta_acc"].append(batch_theta_acc)
    # run["train/epoch/cls_loss"].append(batch_cls_loss)
    # run["train/epoch/cls_loss_theta"].append(batch_cls_loss_theta)
    # run["train/epoch/pair_loss"].append(batch_pair_loss)
    # run["train/epoch/kl_loss"].append(batch_kl_loss.avg)
    # # run["train/epoch/kld_theta"].append(batch_kld_theta)
    # run["train/epoch/recon_loss"].append(batch_recon_loss.avg)
    return iteration_num



def train_cvae_nce(run, config, model, classifier, criterion_cla, criterion_pair, criterion_recon, criterion_nce, criterion_center,
              optimizer, optimizer_center, trainloader, epoch, iteration_num, text_feature_list=None):
    
    if config.DATA.TRAIN_FORMAT == 'novel':
        if config.MODEL.TRAIN_STAGE == 'klNocls_stage':
            model.train()
            model.decoder.eval()
            drawer = tSNE_plot(num_query=None, trainplot=True)
            drawer.reset()
        elif config.MODEL.TRAIN_STAGE == 'reidstage':
            model.eval()
            model.reid_projector.train()
            model.i2t_projector.train()
            classifier.train()
        else:
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
    batch_center_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_kl_loss = AverageMeter()
    batch_kld_theta = AverageMeter()
    batch_recon_loss = AverageMeter()
    batch_loss = AverageMeter()
    batch_acc = AverageMeter()
    batch_i2t_acc = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # acc_meter = AverageMeter()
    i2t_acc_meter = AverageMeter()

    end = time.time()
    # run["train/epoch"].append(epoch)

    for batch_idx, (imgs_tensor, pids, style_ids, data_tag, _) in enumerate(trainloader):
        iteration_num += 1
        # convert fp16 tensor to fp32            
        if not config.TRAIN.AMP:
            imgs_tensor = imgs_tensor.float()

        imgs_tensor, pids, style_ids = imgs_tensor.cuda(), pids.cuda(), style_ids.cuda()

        # run["train/batch/load_time"].append(time.time() - end)
        print("load time: {}".format(time.time() - end))

        '''
        0422 norm or no norm for testing BatchNorm
        '''
        # imgs_tensor = model.norm(imgs_tensor)
        # recon_x, mean, log_var, z, x_pre, x_proj_norm, z_1, theta, logjacobin, domian_feature, flow_input= model.encode(imgs_tensor)   
        # x_pre, z, z_c, z_s, fusez_s, domian_feature, mean, log_var = model.encode(imgs_tensor)

        if config.DATA.DATASET == 'duke' or config.DATA.DATASET == 'msmt17':
            style_ids = torch.zeros_like(style_ids)
            style_ids = style_ids.cuda()
        # styles_onehot = idx2onehot(style_ids, config.MODEL.STYLE_NUM)
        
        x_pre, mean, log_var, z_c, z_s, domian_feature, fusez_s, z, recon_x = model(imgs_tensor, style_ids)
        
        with torch.no_grad():
            text_feature = text_feature_list[pids]

        z_c_reid = model.reid_projector(z_c)
        pair_loss = criterion_pair(z_c_reid, pids)
        center_loss = criterion_center(z_c_reid, pids)

        z_c_proj = model.i2t_projector(z_c_reid)
        i2t_nce_loss = criterion_nce(z_c_proj, text_feature, pids, pids)
        t2i_nce_loss = criterion_nce(text_feature, z_c_proj, pids, pids)
        nce_loss = i2t_nce_loss + t2i_nce_loss
        # Need using all text feature to calculate the logits
        logits = z_c_proj @ text_feature_list.t()
        acc = (logits.max(1)[1] == pids).float().mean()
        i2t_acc_meter.update(acc, 1)

        if 'novel' in config.DATA.TRAIN_FORMAT and config.MODEL.TRAIN_STAGE != 'reidstage':
            drawer.update((z_c, pids, data_tag))
            drawer.update_U(domian_feature)
        

        outputs = classifier(z_c_reid)
        _, preds = torch.max(outputs.data, 1)
        cls_loss_ce = criterion_cla(outputs, pids)

        '''for record'''
        # cls_loss = nce_loss
        cls_loss_i2t = criterion_cla(logits, pids)

        cls_loss = cls_loss_ce + cls_loss_i2t

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
        if config.MODEL.TRAIN_STAGE == 'klNocls_stage':
            loss = recon_loss
            loss = loss + beta * kl_loss
            
            # under NCE, no cls mean no classifier
            # loss = loss + gamma * cls_loss
            # loss = loss + pair_loss
            # loss = loss + center_loss

        elif config.MODEL.TRAIN_STAGE == 'reidstage':
            loss = cls_loss
            # loss = cls_loss_i2t
            loss = loss + pair_loss
            loss = loss + center_loss
        else:
            loss = recon_loss
            loss = loss + beta * kl_loss  # baseline no kl
            # loss = loss + gamma * cls_loss
            loss = loss + gamma * cls_loss
            loss = loss + pair_loss
            # loss = loss + center_loss

        # elif config.MODEL.TRAIN_STAGE == 'reidstage':
        #     loss = cls_loss
        #     # loss = pair_loss
        #     # loss = 0.5 * loss + pair_loss
        
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
        if optimizer_center is not None:
            optimizer_center.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        if optimizer_center is not None:
            optimizer_center.step()

        batch_acc.update((torch.sum(preds == pids.data)).float()/pids.size(0), pids.size(0))
        batch_i2t_acc = i2t_acc_meter
        batch_cls_loss.update(cls_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_center_loss.update(center_loss.item(), pids.size(0))
        batch_kl_loss.update(kl_loss.item(), pids.size(0))
        batch_recon_loss.update(recon_loss.item(), pids.size(0))
        batch_loss.update(loss.item(), pids.size(0))
        batch_time.update(time.time() - end)

        # run['train/batch/1_prior'].append(prior.mean().item())
        # run['train/batch/1_posterior'].append(posterior.mean().item())
        # run["train/batch/cls_loss"].append(cls_loss.item())
        # run["train/batch/pair_loss"].append(pair_loss.item())
        # run['train/batch/center_loss'].append(center_loss.item())
        # run["train/batch/0_kl_loss"].append(kl_loss.item())
        # run["train/batch/0_recon_loss"].append(recon_loss.item())
        # # run["train/batch/0_regular_loss"].append(regular_loss.item())
        # run["train/batch/loss"].append(loss.item())
        # run["train/batch/acc"].append((torch.sum(preds == pids.data)).float()/pids.size(0))
        # run["train/batch/acc_NCE"].append(acc)
        # run["train/batch/batch_time"].append(time.time() - end)
        end = time.time()

    print('Epoch:{0} '
          'Time:{batch_time.sum:.1f} '
          'Data:{data_time.sum:.1f} '
          'Loss:{loss.avg:.4f} '
          'Cls Loss:{cls_loss.avg:.4f} '
          'Pair Loss:{pair_loss.avg:.4f} '
          'Center Loss:{center_loss.avg:.4f} '
          'KL Loss:{kl_loss.avg:.4f} '
          'Recon Loss:{bce_loss.avg:.4f} '
          'Acc:{acc.avg:.4f} '
          'Acc NCE:{acc_meter.avg:.4f}'.format(
            epoch+1, batch_time=batch_time, data_time=data_time, 
            loss=batch_loss, cls_loss=batch_cls_loss,
            pair_loss=batch_pair_loss, center_loss=batch_center_loss,
            kl_loss=batch_kl_loss,
            bce_loss=batch_recon_loss, acc=batch_acc, acc_meter=batch_i2t_acc)
          )
    if 'reid' not in config.MODEL.TRAIN_STAGE:
        if 'novel' in config.DATA.TRAIN_FORMAT:
            if (epoch+1) % 10 == 0:
                print("Jump TSNE")
                # drawer.compute(run)
    # run["train/epoch/loss"].append(batch_loss)
    # run["train/epoch/acc"].append(batch_acc)
    # run["train/epoch/theta_acc"].append(batch_theta_acc)
    # run["train/epoch/cls_loss"].append(batch_cls_loss)
    # run["train/epoch/cls_loss_theta"].append(batch_cls_loss_theta)
    # run["train/epoch/pair_loss"].append(batch_pair_loss)
    # run["train/epoch/kl_loss"].append(batch_kl_loss.avg)
    # # run["train/epoch/kld_theta"].append(batch_kld_theta)
    # run["train/epoch/recon_loss"].append(batch_recon_loss.avg)
    return iteration_num

