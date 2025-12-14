# utils.py
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps, ImageFilter
import random

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    sigma: List of [min, max] radius for blur.
    """
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Solarization(object):
    def __call__(self, x):
        return ImageOps.solarize(x)

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
        ])
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Global crops: 224x224
        # Global 1: Blur p=1.0
        self.global_transfo1 = T.Compose([
            T.RandomResizedCrop(96, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            T.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0), 
            normalize,
        ])
        
        # Global 2: Blur p=0.1, Solarization p=0.2
        self.global_transfo2 = T.Compose([
            T.RandomResizedCrop(96, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1), 
            T.RandomApply([Solarization()], p=0.2),
            normalize,
        ])
        
        # Local crops: 96x96
        # Blur p=0.5
        self.local_transfo = T.Compose([
            T.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5), 
            normalize,
        ])
        self.local_crops_number = local_crops_number

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

def get_transforms(img_size=224):
    train_transform = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform