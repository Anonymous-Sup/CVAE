from torch import nn

from losses.cross_entropy_label_smooth import CrossEntropyLabelSmooth
from losses.triplet_loss import TripletLoss
from losses.contrastive_loss import ContrastiveLoss
from losses.arcface_loss import ArcFaceLoss
from losses.cosface_loss import CosFaceLoss, PairwiseCosFaceLoss
from losses.circle_loss import CircleLoss, PairwiseCircleLoss
from losses.naive_loss_fn import KLD_loss, BCE_loss, MSE_loss, MAE_loss, SmoothL1_loss, Pearson_loss, MMD_loss

def build_losses(config):
    if config.LOSS.CLA_LOSS == "crossentropy":
        criterion_cla = nn.CrossEntropyLoss()
    elif config.LOSS.CLA_LOSS == "crossentropylabelsmooth":
        criterion_cla = CrossEntropyLabelSmooth()
    elif config.LOSS.CLA_LOSS == 'arcface':
        criterion_cla = ArcFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'cosface':
        criterion_cla = CosFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'circle':
        criterion_cla = CircleLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    else:
        raise KeyError("Invalid classification loss: '{}'".format(config.LOSS.CLA_LOSS))

    # Build pairwise loss
    if config.LOSS.PAIR_LOSS == 'triplet':
        criterion_pair = TripletLoss(margin=config.LOSS.PAIR_M, distance=config.TEST.DISTANCE)
    elif config.LOSS.PAIR_LOSS == 'contrastive':
        criterion_pair = ContrastiveLoss(scale=config.LOSS.PAIR_S)
    elif config.LOSS.PAIR_LOSS == 'cosface':
        criterion_pair = PairwiseCosFaceLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'circle':
        criterion_pair = PairwiseCircleLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    else:
        raise KeyError("Invalid pairwise loss: '{}'".format(config.LOSS.PAIR_LOSS))
    
    criterion_kl = KLD_loss()

    if config.LOSS.RECON_LOSS == 'bce':
        criterion_recon = BCE_loss(config.MODEL.FEATURE_DIM)
    elif config.LOSS.RECON_LOSS == 'mse':
        criterion_recon = MSE_loss(config.MODEL.FEATURE_DIM)
    elif config.LOSS.RECON_LOSS == 'mae':
        criterion_recon = MAE_loss(config.MODEL.FEATURE_DIM)
    elif config.LOSS.RECON_LOSS == 'smoothl1':
        criterion_recon = SmoothL1_loss(config.MODEL.FEATURE_DIM)
    elif config.LOSS.RECON_LOSS == 'pearson':
        criterion_recon = Pearson_loss(config.MODEL.FEATURE_DIM)

    criterion_regular = MMD_loss(sigma=1.0)

    return criterion_cla, criterion_pair, criterion_kl, criterion_recon, criterion_regular