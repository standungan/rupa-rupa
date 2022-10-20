from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from configs import configuration

import numpy as np

def train_facedataloader():
    cfg = configuration()
    
    transformer = transforms.Compose([
        transforms.Resize(cfg.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], 
                                std=[0.5,0.5,0.5])
    ])

    facedataset = datasets.ImageFolder(cfg.root_train, transform=transformer)
    facedataloader = DataLoader(facedataset,
                                batch_size=cfg.batch_size, 
                                shuffle=cfg.shuffle,
                                num_workers=cfg.num_workers)
    
    return facedataset, facedataloader

def valid_facedataloader():
    pass

if __name__ == "__main__":
    
    FaceDataset, FaceDataloader = train_facedataloader()
    print(FaceDataset.classes)

    images, target = next(iter(FaceDataloader))
    print(images.shape)
    print(target.shape)
