import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import torch


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def read_tensor(tensor_path):
    """Keep reading tensor until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_tensor = False
    if not osp.exists(tensor_path):
        raise IOError("{} does not exist".format(tensor_path))
    while not got_tensor:
        try:
            tensor = torch.load(tensor_path)
            got_tensor = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(tensor_path))
            pass
    return tensor

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, format_tag, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.tag = format_tag

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, cluster_id = self.dataset[index]
        if self.tag == 'tensor':
            img = read_tensor(img_path)
            if len(img.size()) == 2:
                img = img.squeeze(0)
        else:
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
        return img, pid, camid, cluster_id