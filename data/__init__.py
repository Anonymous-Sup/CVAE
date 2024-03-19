import data.transforms as T
from torch.utils.data import DataLoader
# from data.datasets.market import Market1501
# from data.datasets.cuhk03 import CUHK03
from data.datasets.duke import DukeMTMCreID
# from data.datasets.msmt17 import MSMT17

from data.dataset_loader import ImageDataset
from data.samplers import RandomIdentitySampler


__factory = {
    # 'market1501': Market1501,
    # 'cuhk03': CUHK03,
    'duke': DukeMTMCreID,
    # 'msmt17': MSMT17,
}


def get_names():
    return list(__factory.keys())


def build_dataset(config):
    if config.DATA.DATASET not in __factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(config.DATA.DATASET, __factory.keys()))

    print("Initializing dataset {}".format(config.DATA.DATASET))
    # dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT, split_id=config.DATA.SPLIT_ID,
    #                                          cuhk03_labeled=config.DATA.CUHK03_LABELED, 
    #                                          cuhk03_classic_split=config.DATA.CUHK03_CLASSIC_SPLIT)
    dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT, format_tag=config.DATA.FORMAT_TAG, pretrained=config.MODEL.PRETRAIN, latent_size=config.MODEL.LATENT_SIZE)
    return dataset

def build_transforms(config):
    transform_train = T.Compose([
        T.RandomCroping(config.DATA.HEIGHT, config.DATA.WIDTH, p=config.AUG.RC_PROB),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability = config.AUG.RE_PROB)
    ])

    transform_test = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test


def build_dataloader(config):
    dataset = build_dataset(config)
    if config.DATA.FORMAT_TAG != 'tensor':
        transform_train, transform_test = build_transforms(config)
    else:
        transform_train, transform_test = None, None

    trainloader = DataLoader(ImageDataset(dataset.train, format_tag=config.DATA.FORMAT_TAG, transform=transform_train),
                             sampler=RandomIdentitySampler(dataset.train, num_instances=config.DATA.NUM_INSTANCES),
                             batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                             pin_memory=True, drop_last=True)
    queryloader = DataLoader(ImageDataset(dataset.query, format_tag=config.DATA.FORMAT_TAG, transform=transform_test),
                             batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                             pin_memory=True, drop_last=False, shuffle=False)

    galleryloader = DataLoader(ImageDataset(dataset.gallery, format_tag=config.DATA.FORMAT_TAG, transform=transform_test),
                               batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                               pin_memory=True, drop_last=False, shuffle=False)

    # return trainloader, queryloader, galleryloader, dataset.num_train_pids, dataset.train_centroids, dataset.query_centroids, dataset.gallery_centroids
    return trainloader, queryloader, galleryloader, dataset


def build_singe_test_loader(root_path, pretrained):

    dataset = __factory["duke"](root=root_path, format_tag="tensor", pretrained=pretrained, test_metrix_only=True)
    
    queryloader = DataLoader(ImageDataset(dataset.query, format_tag="tensor", transform=None),
                             batch_size=512, num_workers=4,
                             pin_memory=True, drop_last=False, shuffle=False)

    galleryloader = DataLoader(ImageDataset(dataset.gallery, format_tag="tensor", transform=None),
                               batch_size=512, num_workers=4,
                               pin_memory=True, drop_last=False, shuffle=False)
    return queryloader, galleryloader, dataset