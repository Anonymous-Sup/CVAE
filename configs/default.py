import os
import yaml
from yacs.config import CfgNode as CN


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
_C.MODEL.PRETRAIN = 'CLIP'

_C.MODEL.ENCODER_LAYER_SIZES = [512, 256]
_C.MODEL.LATENT_SIZE = 12
_C.MODEL.DECODER_LAYER_SIZES = [256, 512]
_C.MODEL.FEAT_FUSION = False
# feature dim
_C.MODEL.FEATURE_DIM = 512

# Model path for resuming
_C.MODEL.RESUME = ''

# -----For ResNet--------
# The stride for laery4 in resnet
_C.MODEL.RES4_STRIDE = 1


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

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.MAX_EPOCH = 60
# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adam'
# Learning rate
_C.TRAIN.OPTIMIZER.LR = 0.00035
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 5e-4
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'None'   # 'None', 'MultiStepLR'
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
_C.TEST.EVAL_STEP = 5
# Start to evaluate after specific epoch
_C.TEST.START_EVAL = 0

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Fixed random seed
_C.SEED = 0
# Perform evaluation only
_C.EVAL_MODE = False
# GPU device ids for CUDA_VISIBLE_DEVICES
_C.GPU = '1'
# Path to output folder, overwritten by command line argument
_C.OUTPUT = './outputs'
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

    if args.gpu:
        config.GPU = args.gpu

    if args.train_format:
        config.DATA.TRAIN_FORMAT = args.train_format
    if args.format_tag:
        config.DATA.FORMAT_TAG = args.format_tag

    if args.resume:
        config.MODEL.RESUME = args.resume
    
    if args.eval:
        config.EVAL_MODE = True
    
    if args.tag:
        config.TAG = args.tag

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.DATA.DATASET, config.TAG)

    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)

    return config