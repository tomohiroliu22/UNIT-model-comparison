import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def get_transform(grayscale=False, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.root_A = root+"/%sA" % mode + "/"
        self.root_B = root+"/%sB" % mode + "/"
        self.files_A = sorted(os.listdir(self.root_A))
        self.files_B = sorted(os.listdir(self.root_B))
        self.transform_A = get_transform(grayscale=False)
        self.transform_B = get_transform(grayscale=True)

    def __getitem__(self, index):
        image_A = Image.open(self.root_A+self.files_A[index % len(self.files_A)]).convert("RGB")
        image_B = Image.open(self.root_B+self.files_B[index % len(self.files_B)]).convert("RGB")
        # Convert grayscale images to rgb
        item_A = self.transform_A(image_A)
        item_B = self.transform_B(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
