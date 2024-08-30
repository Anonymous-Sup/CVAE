import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import shutil
from configs.default import get_config
from data import build_dataloader
from models import build_model
from losses import build_losses
from train import train_cvae, train_cvae_nce
from test import test_cvae, test_clip_feature
from tools.eval_metrics import evaluate
from tools.utils import AverageMeter, save_checkpoint, set_seed, mkdir_if_missing
from torch.cuda.amp import GradScaler, autocast
import neptune
from utils import EarlyStopping
from scipy.io import loadmat

# torch.autograd.set_detect_anomaly(True)

run = neptune.init_run(
    project="Zhengwei-Lab/MayCVAE",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2ODIwNTQ4Yy0xZDA3LTRhNDctOTRmMy02ZjRlMmMzYmYwZjUifQ==",
)

def parse_option():
    parser = argparse.ArgumentParser(description='Train CVAE model for transfer learning')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, default='duke', help="duke, market1k")
    parser.add_argument('--format_tag', type=str, choices=['tensor', 'img'], help="Using image or pretrained features")
    
    # Training
    parser.add_argument('--train_format', type=str, required= True, choices=['base', 'novel', 'novel_train_from_scratch'], help="Select the datatype for training or finetuning")
    parser.add_argument('--train_stage', type=str, choices=['klstage', 'klNocls_stage', 'kl_cls_stage', 'kl_reid_stage', 'CLSstage', 'reidstage'], required=True, help="Select the stage for training")

    # Parameters 
    parser.add_argument('--vae_type', type=str, choices=['cvae','SinpleVAE'], help="Type of VAE model")
    parser.add_argument('--flow_type', type=str, choices=['Planar', 'Radial', 'RealNVP', 'invertmlp', "yuke_mlpflow"], help="Type of flow model")
    parser.add_argument('--recon_loss', type=str, choices=['bce', 'mse', 'mae', 'smoothl1', 'pearson'], help="Type of reconstruction loss")
    parser.add_argument('--reid_loss', type=str, choices=['crossentropy', 'crossentropylabelsmooth', 'arcface', 'cosface', 'circle'], help="Type of reid loss")
    parser.add_argument('--gaussian', type=str, choices=['Normal', 'MultivariateNormal'], help="Type of gaussion distribution")
    parser.add_argument('--use_NCE', action='store_true', help="Use NCE loss for training")
    parser.add_argument('--use_two_encoder', action='store_true', help="Use 2 encoders models for training")
    # config.MODEL.PROJECTION_TYPE
    parser.add_argument('--projection_type', type=str, choices=['Linear+CLS', 'MLP+CLS', 'Linear1280+CLS', 'MLP1280+CLS', 'MLP768+CLS', 'Transforer1280+CLS'], help="Type of projection head")
    # debug
    parser.add_argument('--only_x_input', action='store_true', help="Use only x as input for flow model")
    parser.add_argument('--only_cvae_kl', action='store_true', help="Use orginal kl loss for cvae model")
    parser.add_argument('--use_centroid', action='store_true', help="Use centroid as domain index")

    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--saved_name', type=str, required=True, help="your output name to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--amp', action='store_true', help="automatic mixed precision")
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    param = {
        'saved_name': args.saved_name,
        'train_stage': args.train_stage,
        'train_format': args.train_format,
        'vae_type': config.MODEL.VAE_TYPE,
        "optimizer": config.TRAIN.OPTIMIZER.NAME,
        "lr": config.TRAIN.OPTIMIZER.LR,
        "weight_decay": config.TRAIN.OPTIMIZER.WEIGHT_DECAY,
        "scheduler": config.TRAIN.LR_SCHEDULER.NAME,
        "flow_type": config.MODEL.FLOW_TYPE,
        "recon_loss": config.LOSS.RECON_LOSS,
        "step_size": config.TRAIN.LR_SCHEDULER.STEPSIZE,
        "decay_rate": config.TRAIN.LR_SCHEDULER.DECAY_RATE,
        "batch_size": config.DATA.TRAIN_BATCH,
        "epoch": config.TRAIN.MAX_EPOCH,
        "seed": config.SEED,
    }
    run["parameters"] = param
    return config

def main(config):
    # Build dataloader
    trainloader, queryloader, galleryloader, dataset = build_dataloader(config)

    # Build model 
    model, classifier, classifier_reID = build_model(config, dataset.num_train_pids)

    # Build loss
    criterion_cla, criterion_pair, criterion_kl, criterion_recon, criterion_nce, criterion_circle = build_losses(config, dataset.num_train_pids)

    early_stopping = EarlyStopping(patience=100, threshold=1.0)

    # Build optimizer
    # select parameters beside the FLOWs parameters in the model
    parameters = []

    for name, param in model.named_parameters():
        if config.DATA.TRAIN_FORMAT == 'novel':
            if config.MODEL.TRAIN_STAGE == 'klNocls_stage':
                if 'reid_projector' in name or 'i2t_projector' in name or 'decoder' in name:
                    param.requires_grad = False
                else:
                    parameters.append(param)
            elif config.MODEL.TRAIN_STAGE == 'kl_cls_stage':
                if 'reid_projector' in name or 'decoder' in name:
                    param.requires_grad = False
                else:
                    parameters.append(param)
            elif config.MODEL.TRAIN_STAGE == 'kl_reid_stage':
                if 'i2t_projector' in name or 'decoder' in name:
                    param.requires_grad = False
                else:
                    parameters.append(param)
            else:
                if 'decoder' in name:
                    param.requires_grad = False
                else:
                    parameters.append(param)
        else: # for base data
            if config.MODEL.TRAIN_STAGE == 'klNocls_stage':
                if 'reid_projector' in name or 'i2t_projector' in name:
                    param.requires_grad = False
                else:
                    parameters.append(param)
            elif config.MODEL.TRAIN_STAGE == 'kl_cls_stage':
                if 'reid_projector' in name:
                    param.requires_grad = False
                else:
                    parameters.append(param)
            elif config.MODEL.TRAIN_STAGE == 'kl_reid_stage':
                if 'i2t_projector' in name:
                    param.requires_grad = False
                else:
                    parameters.append(param)
            else:
                parameters.append(param)
        


    cla_parameters = list(classifier.parameters()) + list(classifier_reID.parameters())

    if config.DATA.TRAIN_FORMAT == 'novel':
        alpha_lr = 1.0   # base lr 1e-4, classifier lr 1e-3
    else:
        alpha_lr = 1.0
    
    i2t_parameters = []
    reid_parameters = []
    if config.TRAIN.OPTIMIZER.NAME == 'adam':
        # use adam that set different learning rate for different parameters
        if config.MODEL.TRAIN_STAGE == 'reidstage':
            if config.LOSS.USE_NCE:
                for name, param in model.named_parameters():
                    if 'i2t_projector' in name:
                        param.requires_grad = True
                        print("{} is tuneable".format(name))
                        i2t_parameters.append(param)
                    elif 'reid_projector' in name:
                        param.requires_grad = True
                        print("{} is tuneable".format(name))
                        reid_parameters.append(param)
                    else:
                        param.requires_grad = False
                all_tuned_parameters = i2t_parameters + reid_parameters + cla_parameters
                optimizer = optim.Adam(all_tuned_parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                                weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
                optimizer_center = optim.SGD(criterion_circle.parameters(), lr=0.5)
            else:
                for name, param in model.named_parameters():
                    if 'reid_projector' in name:
                        param.requires_grad = True
                        print("{} is tuneable".format(name))
                        reid_parameters.append(param)
                for cls_param in cla_parameters:
                    cls_param.requires_grad = False

                optimizer = optim.Adam(reid_parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                            weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
                optimizer_center = optim.SGD(criterion_circle.parameters(), lr=0.5)
        elif config.MODEL.TRAIN_STAGE == 'CLSstage':
            for name, param in model.named_parameters():
                if 'i2t_projector' in name:
                    param.requires_grad = True
                    print("{} is tuneable".format(name))
                    i2t_parameters.append(param)
                else:
                    param.requires_grad = False
            optimizer = optim.Adam([ 
                {'params': filter(lambda p: p.requires_grad ,parameters)},
                {'params': filter(lambda p: p.requires_grad ,cla_parameters), 'lr': config.TRAIN.OPTIMIZER.LR * alpha_lr}], 
                lr=config.TRAIN.OPTIMIZER.LR, weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
            optimizer_center = None
        elif config.MODEL.TRAIN_STAGE == 'klNocls_stage':
            for cls_param in cla_parameters:
                cls_param.requires_grad = False
            optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                                weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
            optimizer_center = None
        elif config.MODEL.TRAIN_STAGE == 'kl_reid_stage':
            for cls_param in cla_parameters:
                cls_param.requires_grad = False
            optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR,
                                weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
            optimizer_center = optim.SGD(criterion_circle.parameters(), lr=0.5)
        else:
            optimizer = optim.Adam([
            {'params': filter(lambda p: p.requires_grad ,parameters)},
            {'params': filter(lambda p: p.requires_grad ,cla_parameters), 'lr': config.TRAIN.OPTIMIZER.LR * alpha_lr}], 
            lr=config.TRAIN.OPTIMIZER.LR, weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)

            optimizer_center = optim.SGD(criterion_circle.parameters(), lr=0.5)

        # optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
        #                        weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        
    elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD([
                {'params': filter(lambda p: p.requires_grad ,parameters)},
                {'params': filter(lambda p: p.requires_grad ,cla_parameters), 'lr': config.TRAIN.OPTIMIZER.LR * alpha_lr}], 
                lr=config.TRAIN.OPTIMIZER.SGDLR, momentum=0.9, weight_decay=config.TRAIN.OPTIMIZER.SGD_WEIGHT_DECAY, nesterov=True)
    else:
        raise KeyError("Unknown optimizer: {}".format(config.TRAIN.OPTIMIZER.NAME))
    
    # Custom warm-up function
    def warmup_lr_lambda(current_epoch, warmup_epochs, regualr_scheduler):
        if current_epoch < warmup_epochs:
            return float(current_epoch+1) / float(warmup_epochs)
        else:
            return regualr_scheduler.get_last_lr()[0] / regualr_scheduler.base_lrs[0]

    # Build lr_scheduler
    if config.TRAIN.LR_SCHEDULER.NAME != 'None':
        # main scheduler
        main_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, 
                                            gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)
        if optimizer_center is not None:
            scheduler_center = lr_scheduler.MultiStepLR(optimizer_center, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, 
                                            gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)
        warmup_epochs = 5
        # Combined scheduler using LambdaLR
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: warmup_lr_lambda(epoch, warmup_epochs, main_scheduler))
        
    if config.TRAIN.AMP:
        scaler = GradScaler()

    start_epoch = config.TRAIN.START_EPOCH
    
    best_rank1 = -np.inf
    best_mAP = -np.inf
    best_acc = [-np.inf, -np.inf, -np.inf]

    if config.EVAL_MODE:
        print("Overwrite the checkpoint for evaluation")
        print("Loading checkpoint from '{}/{}'".format(config.MODEL.RESUME, 'best_model.pth.tar'))
        checkpoint = torch.load(config.MODEL.RESUME + '/best_model.pth.tar')
        model.load_param(checkpoint['model'], ignore_i2t=False, ignore_reid=False)
        # flows_model.load_state_dict(checkpoint['flows_model'])
        classifier.load_state_dict(checkpoint['classifier'])
        if "classifier_reID" in checkpoint:
            classifier_reID.load_state_dict(checkpoint['classifier_reID'])
        start_epoch = checkpoint['epoch']
        best_rank1 = checkpoint['rank1']
        del checkpoint
    else:
        if config.DATA.TRAIN_FORMAT == "novel":
            if 'kl' in config.MODEL.TRAIN_STAGE:
                print("=> Start tuning the model on Novel data")
                print("Loading checkpoint from '{}/{}'".format(config.MODEL.RESUME, 'best_model.pth.tar'))
                checkpoint = torch.load(config.MODEL.RESUME + '/best_model.pth.tar')
                # ignore_i2t means that the i2t_projector is not loaded
                model.load_param(checkpoint['model'], ignore_i2t=False, ignore_reid=False)
                print("orginal best rank1 = {}".format(checkpoint['rank1']))
                # flows_model.load_state_dict(checkpoint['flows_model'])
                del checkpoint
                # copy the checkpoint to the output folder
                print("=> Copy the checkpoint to the output folder")
                base_folder = os.path.basename(config.MODEL.RESUME)
                output_file = os.path.join(config.OUTPUT, base_folder)
                mkdir_if_missing(output_file)
                shutil.copy(config.MODEL.RESUME + '/best_model.pth.tar', output_file)
            else:
                print("=> Start Training ReID projector and Classifier")
                print("Loading checkpoint from '{}/{}'".format(config.MODEL.RESUME, 'best_model.pth.tar'))
                checkpoint = torch.load(config.MODEL.RESUME + '/best_model.pth.tar')
                model.load_param(checkpoint['model'], ignore_i2t=False, ignore_reid=False)
                classifier.load_state_dict(checkpoint['classifier'])
                print("orginal best rank1 = {}".format(checkpoint['rank1']))
                del checkpoint

        elif config.DATA.TRAIN_FORMAT == "novel_train_from_scratch":
            print("=> Start training the model on Novel data from scratch")
        else: # base data
            if 'kl' not in config.MODEL.TRAIN_STAGE:
                print("=> Start Training second stage model")
                print("Loading checkpoint from '{}/{}'".format(config.MODEL.RESUME, 'best_model.pth.tar'))
                checkpoint = torch.load(config.MODEL.RESUME + '/best_model.pth.tar')
                model.load_param(checkpoint['model'], ignore_i2t=False, ignore_reid=False)
                classifier.load_state_dict(checkpoint['classifier'])
                print("orginal best rank1 = {}".format(checkpoint['rank1']))
                del checkpoint
                # flows_model.load_state_dict(checkpoint['flows_model'])
            else:
                if config.MODEL.RESUME:
                    print("Loading checkpoint from '{}/{}'".format(config.MODEL.RESUME, 'best_model.pth.tar'))
                    checkpoint = torch.load(config.MODEL.RESUME + '/best_model.pth.tar')
                    model.load_param(checkpoint['model'])
                    # flows_model.load_state_dict(checkpoint['flows_model'])
                    classifier.load_state_dict(checkpoint['classifier'])
                    start_epoch = checkpoint['epoch']
                    best_rank1 = checkpoint['rank1']
                    del checkpoint
        
    # Set device
    model = model.cuda()
    # flows_model = flows_model.cuda()
    classifier = classifier.cuda()
    classifier_reID = classifier_reID.cuda()

    if config.LOSS.USE_NCE:
        
        text_path_128 = '/home/zhengwei/Desktop/Zhengwei/Projects/datasets/Market-Sketch-1K/tensor/CLIPreidNew/textemb/2sketch_tune_linear/text_features.mat'
        text_path_512 = '/home/zhengwei/Desktop/Zhengwei/Projects/datasets/Market-Sketch-1K/tensor/CLIPreidFinetune/textemb/2sketch_tune/text_features.mat'
        
        test_base_duke = '/home/zhengwei/Desktop/Zhengwei/Projects/datasets/DukeMTMC-reID/tensor/CLIPreidNew/textemb/base_duke/text_features.mat'
        
        text_tune_sysu = '/home/zhengwei/Desktop/Zhengwei/Projects/datasets/sysu_mm_01/fewshot_label/tensor/CLIPreidFinetune/textemb/tune_sysumm01/text_features.mat'
        if config.DATA.DATASET == 'market1k':
            results = loadmat(text_path_512)
        elif config.DATA.DATASET == 'duke':
            results = loadmat(test_base_duke)
        elif config.DATA.DATASET == 'sysu_mm01':
            results = loadmat(text_tune_sysu)
        else:
            raise KeyError("Unknown Text embedding for: {}".format(config.DATA.DATASET))

        text_embeddings = torch.tensor(results['text_features']).float().cuda()
        del results
    else:
        text_embeddings = None
    
    if config.EVAL_MODE or config.DATA.TRAIN_FORMAT == 'novel' or 'kl' not in config.MODEL.TRAIN_STAGE:
        if config.DATA.TRAIN_FORMAT == 'novel':
            print("=> Start evaluation on Novel data without finetuning")
        else:
            print("=> Start evaluation only ")
        with torch.no_grad():
            print("=> Test pretarined feature form VLP model")
            test_clip_feature(queryloader, galleryloader, config.DATA.DATASET)
            test_cvae(run, config, model, queryloader, galleryloader, dataset, classifier, classifier_reID, text_embeddings, latent_z='new_z')
            if config.EVAL_MODE:
                final_epoch = True
            else:
                final_epoch = False
            test_cvae(run, config, model, queryloader, galleryloader, dataset, classifier, None, text_embeddings, latent_z='z_c', final_epoch=False)

        if config.EVAL_MODE:
            return

    start_time = time.time()
    train_time = 0
    best_epoch = 0
    iteration_num = 0
    print("=> Start training")
    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
        start_train_time = time.time()
        if config.TRAIN.AMP:
            iteration_num = train_cvae(run, config, model, classifier, criterion_cla, criterion_pair, criterion_kl, criterion_recon, criterion_regular,
              optimizer, trainloader, epoch, dataset.train_centroids, early_stopping, scaler)
        else:
            if config.LOSS.USE_NCE:
                iteration_num = train_cvae_nce(run, config, model, classifier, classifier_reID, criterion_cla, criterion_pair, criterion_recon, criterion_nce, criterion_circle,
                optimizer, optimizer_center, trainloader, epoch, iteration_num, text_embeddings)
            else:
                iteration_num = train_cvae(run, config, model, classifier, classifier_reID, criterion_cla, criterion_pair, criterion_recon, criterion_circle,
                optimizer, optimizer_center, trainloader, epoch, iteration_num)
            # for name, param in classifier.named_parameters():
            #     print(f'Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n')
        train_time += round(time.time() - start_train_time)
        
        if (epoch+1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 and \
            (epoch+1) % config.TEST.EVAL_STEP == 0 or (epoch+1) == config.TRAIN.MAX_EPOCH:
            
            print("=> Test at epoch {}".format(epoch+1))
            with torch.no_grad():
                test_cvae(run, config, model, queryloader, galleryloader, dataset, classifier, classifier_reID, text_embeddings, latent_z='z_c')
                rank, mAP, acc_total = test_cvae(run, config, model, queryloader, galleryloader, dataset, classifier, classifier_reID, text_embeddings, latent_z='new_z')
                # test_cvae(None, config, model, queryloader, galleryloader, dataset, classifier, latent_z='x_pre')
                # test_cvae(None, config, model, queryloader, galleryloader, dataset, classifier, latent_z='mu')
                
                # run["eval/rank1"].append(rank1)
                rank1 = rank[0]

            is_best = (rank1 + mAP + acc_total[2]) > (best_rank1 + best_mAP + best_acc[2])
            
            if is_best: 
                best_rank1 = rank1
                best_cmc = rank
                best_mAP = mAP
                best_acc = acc_total
                best_epoch = epoch + 1
            
            if (epoch+1) == config.TRAIN.MAX_EPOCH:
                final_epoch = True
            else:
                final_epoch = False
            
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'classifier': classifier.state_dict(),
                'classifier_reID': classifier_reID.state_dict(),
                'cmc': best_cmc,
                'acc': best_acc,
                'mAP': best_mAP,
                'rank1': rank1,
                'best_epoch': best_epoch,
                'optimizer': optimizer.state_dict(),
            }, is_best, final_epoch, osp.join(config.OUTPUT, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
        
        
        # Function to get the current learning rate
        def get_current_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']
        
        if config.TRAIN.LR_SCHEDULER.NAME != 'None':
            run['train/lr'].append(get_current_lr(optimizer))
            main_scheduler.step()
            scheduler.step()
            if optimizer_center is not None:
                scheduler_center.step()
            
            
    print("=> Best Rank-1 {:.1%}, mAP {:.1%} achieved at epoch {}".format(best_rank1, best_mAP, best_epoch))
    run["best_rank1"] = best_rank1
    run['best_mAP'] = best_mAP
    run["best_epoch"] = best_epoch
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    gc.collect()
    
    config = parse_option()
    # set gpu from '0,1' to '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU

    set_seed(config.SEED)
    print("=> Configurations:\n-------------------------")
    print(config)
    print("----------------------")
    # run['parameters'] = config
    main(config)
    run.stop()
