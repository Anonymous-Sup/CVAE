import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from configs.default import get_config
from data import build_dataloader
from models import build_model
from losses import build_losses
from train import train_cvae
from test import test_cvae
from tools.eval_metrics import evaluate
from tools.utils import AverageMeter, save_checkpoint, set_seed

import neptune
run = neptune.init_run(
    project="Zhengwei-Lab/NIPSTransferReID",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2ODIwNTQ4Yy0xZDA3LTRhNDctOTRmMy02ZjRlMmMzYmYwZjUifQ==",
)  # your credentials

def parse_option():
    parser = argparse.ArgumentParser(description='Train CVAE model for transfer learning')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, default='duke', help="duke, market1501, cuhk03, msmt17")
    parser.add_argument('--format_tag', type=str, choices=['tensor', 'img'], help="Using image or pretrained features")
    parser.add_argument('--train_format', type=str, required= True, choices=['base', 'normal'], help="Select the datatype for training or finetuning")
    
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--amp', action='store_true', help="automatic mixed precision")
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config

def main(config):
    # Build dataloader
    trainloader, queryloader, galleryloader, dataset = build_dataloader(config)

    # Build model 
    model, classifier = build_model(config, dataset.num_train_pids)

    # Build loss
    criterion_cla, criterion_pair, criterion_kl, criterion_bce = build_losses(config)

    # Build optimizer
    parameters = list(model.parameters()) + list(classifier.parameters())
    
    if config.TRAIN.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, 
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
    else:
        raise KeyError("Unknown optimizer: {}".format(config.TRAIN.OPTIMIZER.NAME))
    
    # Build lr_scheduler
    if config.TRAIN.LR_SCHEDULER.NAME != 'None':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, 
                                            gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)
    
    start_epoch = config.TRAIN.START_EPOCH
    
    if config.MODEL.RESUME:
        print("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME)
        model.load_state_dict(checkpoint['model'])
        classifier.load_state_dict(checkpoint['classifier'])
        start_epoch = checkpoint['epoch']
    
    if config.DATA.TRAIN_FORMAT != "base":
        print("=> Start Finetuning the model")
        print("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME)
        model.load_state_dict(checkpoint['model'])
        
    # Set device
    model = model.cuda()
    classifier = classifier.cuda()

    if config.EVAL_MODE:
        print("=> Start evaluation only ")
        with torch.no_grad():
            test_cvae(run, config, model, queryloader, galleryloader, dataset)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("=> Start training")
    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
        start_train_time = time.time()
        train_cvae(run, config, model, classifier, criterion_cla, criterion_pair, criterion_kl, criterion_bce, 
              optimizer, trainloader, epoch, dataset.train_centroids)
        train_time += round(time.time() - start_train_time)
        
        
        if (epoch+1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 and \
            (epoch+1) % config.TEST.EVAL_STEP == 0 or (epoch+1) == config.TRAIN.MAX_EPOCH:
            print("=> Test at epoch {}".format(epoch+1))
            with torch.no_grad():
                rank1 = test_cvae(run, config, model, queryloader, galleryloader, dataset)
            
            is_best = rank1 > best_rank1
            
            if is_best: 
                best_rank1 = rank1
                best_epoch = epoch + 1
            
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'classifier': classifier.state_dict(),
                'rank1': rank1,
                'optimizer': optimizer.state_dict(),
            }, is_best, osp.join(config.OUTPUT, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
        
        if config.TRAIN.LR_SCHEDULER.NAME != 'None':
            scheduler.step()
    print("=> Best Rank-1 {:.1%} achieved at epoch {}".format(best_rank1, best_epoch))
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = parse_option()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    set_seed(config.SEED)
    print("=> Configurations:\n-------------------------")
    print(config)
    print("----------------------")
    run['parameters'] = config
    main(config)
    run.stop()
