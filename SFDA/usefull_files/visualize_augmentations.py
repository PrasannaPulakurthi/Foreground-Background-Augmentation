import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.utils import make_grid
from moco.loader import NCropsTransform
from utils import get_augmentation
from image_list import ImageList
import random
import numpy as np
import torch
from torchvision.utils import save_image

# Function to set seed values
def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def unnormalize(tensor, mean, std):
    """Unnormalize a tensor image."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Augmentations
def get_augmentation_versions_patches(name):
    transform_list = [
        get_augmentation("test"),
        get_augmentation("mask"),
        get_augmentation("jigsaw", patch_height=28, mix_prob=1),
        get_augmentation("ours_1", patch_height=28, mix_prob=1),
        get_augmentation("ours", alpha=4.0, beta=2.0, patch_height=28, mix_prob=1), # 8x8
    ]
    return NCropsTransform(transform_list)


def main(train_transform, folder_name):
    # Check if the folder exists; if not, create it
    if not os.path.exists(f"output/visualize/{folder_name}"):
        os.makedirs(f"output/visualize/{folder_name}")

    # Example usage with the specified replacements
    image_root = 'datasets/PACS'  # Replace args.data.image_root
    pseudo_item_list = None  # Replace pseudo_item_list
    batch_size = 1  # Replace args.data.batch_size
    num_workers = 1  # Replace args.data.workers
    label_file = 'datasets/PACS/photo_list.txt'

    # Training data
    train_dataset = ImageList(
        image_root=image_root,
        label_file=label_file,  # uses pseudo labels
        transform=train_transform,
        pseudo_item_list=pseudo_item_list,
    )
    train_sampler = None  # Assuming single-process training, no distributed sampler needed
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False,
    )

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Save images during the training loop
    for i, data in enumerate(train_loader):
        # unpack and move data
        images, _, idxs = data
        images = [unnormalize(img.cpu(), mean, std) for img in images]
        # Iterate over each image and save it with the index as the filename
        for j, image in enumerate(images):
            # Move the image tensor to CUDA and save it with the index as the filename
            image = image.to("cuda")
            save_image(image, f"output/visualize/{folder_name}/image_{i}_{j}.png")
        if i>50:
            break

if __name__ == '__main__':
    # Set seed for reproducibility
    set_seed(69)

    # Different patch sizes
    train_transform = get_augmentation_versions_patches("ours")
    main(train_transform,"ours")