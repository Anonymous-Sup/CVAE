import os
import yaml
from yacs.config import CfgNode as CN
import datetime

_C = CN()
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Root path for dataset directory
_C.DATA.ROOT = '/home/zhengwei/github/datasets'
# Dataset for evaluation
_C.DATA.DATASET = 'duke'
# Dataset format, using CLIP pretrained feautre
_C.DATA.FORMAT_TAG = 'tensor'
_C.DATA.TRAIN_FORMAT = 'base'

# Workers for dataloader
_C.DATA.NUM_WORKERS = 4
# Batch size for training
_C.DATA.TRAIN_BATCH = 64
# Batch size for testing
_C.DATA.TEST_BATCH = 512
# The number of instances per identity for training sampler
_C.DATA.NUM_INSTANCES = 4

# -------For Image -----------
# Height of input image
_C.DATA.HEIGHT = 256
# Width of input image
_C.DATA.WIDTH = 128

# -------For CUHK03-----------
# Split index for CUHK03
_C.DATA.SPLIT_ID = 0
# Whether to use labeled images, if false, detected images are used
_C.DATA.CUHK03_LABELED = False
# Whether to use classic split by Li et al. CVPR'14 (default: False)
_C.DATA.CUHK03_CLASSIC_SPLIT = False

# -----------------------------------------------------------------------------
# Default Augmentation settings for Image
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Random crop prob
_C.AUG.RC_PROB = 0.5
# Random erase prob
_C.AUG.RE_PROB = 0.5


# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = 'CVAE'
_C.MODEL.PRETRAIN = 'CLIPreid'
_C.MODEL.VAE_TYPE = 'CVAE'
_C.MODEL.FLOW_TYPE = 'Planar'  # 'Planar', 'Radial', 'RealNVP'
# _C.MODEL.ENCODER_LAYER_SIZES = [1280, 256]
_C.MODEL.LATENT_SIZE = 12
# _C.MODEL.DECODER_LAYER_SIZES = [256, 1280]
_C.MODEL.FEAT_FUSION = True
# feature dim
_C.MODEL.FEATURE_DIM = 1280

# Model path for resuming
_C.MODEL.RESUME = ''
_C.MODEL.TRAIN_STAGE = '' # 'klstage', 'reidstage'

# -----For ResNet--------
# The stride for laery4 in resnet
_C.MODEL.RES4_STRIDE = 1


# ----For debug-----
_C.MODEL.ONLY_X_INPUT = False
_C.MODEL.ONLY_CVAE_KL = False
_C.MODEL.USE_CENTROID = False
_C.MODEL.CONDITIONAL = False

# -----------------------------------------------------------------------------
# Losses for training 
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# Classification loss
_C.LOSS.CLA_LOSS = 'crossentropy'
# Scale
_C.LOSS.CLA_S = 16.
# Margin
_C.LOSS.CLA_M = 0.

# Pairwise loss
_C.LOSS.PAIR_LOSS = 'triplet'
# Scale
_C.LOSS.PAIR_S = 16.
# Margin
_C.LOSS.PAIR_M = 0.3

_C.LOSS.RECON_LOSS = 'mse' # 'bce', 'mse', 'mae', 'smoothl1', 'pearson'

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.MAX_EPOCH = 60
_C.TRAIN.AMP = False

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adam'
# Learning rate
_C.TRAIN.OPTIMIZER.LR = 1e-2  # ori: 0.00035
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4   # ori: 5e-4
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'MultiStepLR'   # 'None', 'MultiStepLR'
# Stepsize to decay learning rate
_C.TRAIN.LR_SCHEDULER.STEPSIZE = [20, 40]
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Similarity for testing
_C.TEST.DISTANCE = 'cosine'
# Perform evaluation after every N epochs (set to -1 to test after training)
_C.TEST.EVAL_STEP = 5   # 5
# Start to evaluate after specific epoch
_C.TEST.START_EVAL = 0

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Fixed random seed
_C.SEED = 0
# 
_C.AMP = True
# Perform evaluation only
_C.EVAL_MODE = False
# GPU device ids for CUDA_VISIBLE_DEVICES
_C.GPU = '1'
# Path to output folder, overwritten by command line argument
_C.OUTPUT = './outputs'
_C.SAVED_NAME = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'CVAE_default'


def update_config(config, args):
    config.defrost()

    print('=> merge config from {}'.format(args.cfg))
    config.merge_from_file(args.cfg)

    # merge from specific arguments
    if args.root:
        config.DATA.ROOT = args.root
    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.output:
        config.OUTPUT = args.output
    
    if args.saved_name:
        config.SAVED_NAME = args.saved_name
    
    if args.vae_type:
        config.MODEL.VAE_TYPE = args.vae_type
    if args.flow_type:
        config.MODEL.FLOW_TYPE = args.flow_type
    
    if args.recon_loss:
        config.LOSS.RECON_LOSS = args.recon_loss

    if args.gpu:
        config.GPU = args.gpu

    if args.train_format:
        config.DATA.TRAIN_FORMAT = args.train_format
    if args.format_tag:
        config.DATA.FORMAT_TAG = args.format_tag

    if args.only_x_input:
        print("Use only x as input for flow model")
        config.MODEL.ONLY_X_INPUT = True

    if args.only_cvae_kl:
        print("Use original kl loss for cvae model")
        config.MODEL.ONLY_CVAE_KL = True

    if args.use_centroid:
        print("Use centroid as domain embedding")
        config.MODEL.USE_CENTROID = True
    # if args.conditionalvae:
    #     print("Use conditional vae, instead of additional domain embedding")
    #     config.MODEL.CONDITIONAL = True

    if args.resume:
        config.MODEL.RESUME = args.resume

    if args.train_stage:
        config.MODEL.TRAIN_STAGE = args.train_stage
        if 'reid' in args.train_stage:
            if not args.resume:
                raise ValueError("Need to specify the model path for resuming")

    if args.eval:
        config.EVAL_MODE = True
    
    if args.tag:
        config.TAG = args.tag

    if args.amp:
        config.TRAIN.AMP = args.amp

    datetime_today = str(datetime.date.today())
    # output folder
    if 'reid' in args.train_stage:
        config.OUTPUT = os.path.join(config.MODEL.RESUME, 'reid')
    else:
        config.OUTPUT = os.path.join(config.OUTPUT, config.DATA.DATASET, config.TAG, datetime_today, config.SAVED_NAME + "_" + config.MODEL.FLOW_TYPE+"_"+config.LOSS.RECON_LOSS)

    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)

    return config