# -*- coding: utf-8 -*-

import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
from tqdm import tqdm
import argparse
from augmentations import RandomErasing, RandomGrayscaleErasing, RandomPatchNoise, Ours
from train import PairedImageFolder

# Argument Parser
parser = argparse.ArgumentParser(description='Preprocess and Save Augmented Images with Unnormalization')

# Data options
parser.add_argument('--data_dir', default='data/Market/pytorch', type=str, help='Dataset path')
parser.add_argument('--target_folder', default='./processed_images', type=str, help='Target save directory')
parser.add_argument('--train_all', action='store_true', help='Use all training data')  # New argument

# Augmentations
parser.add_argument('--color_jitter', action='store_true', help='Apply color jitter')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability')
parser.add_argument('--bps', default=0, type=float, help='Our version of Random Erasing probability')
parser.add_argument('--rge', default=0, type=float, help='Random Grayscale Erasing probability in [0,1]')
parser.add_argument('--rpe', default=0, type=float, help='Random Patch Erase probability')
parser.add_argument('--rpn', default=0, type=float, help='Random Patch Noise probability')
parser.add_argument('--ours', default=0, type=float, help='Ours augmentation probability')
parser.add_argument('--use_PCB', action='store_true', help='Use PCB-based transformations')
parser.add_argument('--use_swin', action='store_true', help='Use Swin Transformer with 224x224 resolution')
parser.add_argument('--batchsize', default=32, type=int, help='Batch size for data loading')

opt = parser.parse_args()

# Define Image Transformations returning a dict
def get_transforms():
    if opt.use_swin:
        h, w = 224, 224
    elif opt.use_PCB:
        h, w = 384, 192
    else:
        h, w = 256, 128

    # Training transform for images (with augmentations)
    train_transform_list = [
        transforms.Resize((h, w), interpolation=3),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # Validation transform for masks (or reference images) – typically no random flip/jitter
    val_transform_list = [
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    if opt.color_jitter:
        train_transform_list.insert(0, transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))
    if opt.rge > 0:
        train_transform_list.append(RandomGrayscaleErasing(probability=opt.rge))
    if opt.erasing_p > 0:
        train_transform_list.append(RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0]))
    if opt.rpn > 0:
        train_transform_list.append(RandomPatchNoise(probability=opt.rpn))
    return {
        'train': transforms.Compose(train_transform_list),
        'val': transforms.Compose(val_transform_list)
    }

# Unnormalization transformation
def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean  # Reverse normalization
    tensor = torch.clamp(tensor, 0, 1)  # Clamp to valid image range
    return tensor

# Load Dataset and create DataLoaders using the new PairedImageFolder
def load_data():
    data_transforms = get_transforms()
    train_folder = os.path.join(opt.data_dir, 'train' + ('_all' if opt.train_all else ''))
    
    image_datasets = {}
    image_datasets['train'] = PairedImageFolder(
        images_root=train_folder,
        masks_root=train_folder,
        image_transform=data_transforms['train'],
        mask_transform=data_transforms['val']
    )
    image_datasets['val'] = datasets.ImageFolder(
        os.path.join(opt.data_dir, 'val'),
        data_transforms['val']
    )

    dataloaders = {
        phase: torch.utils.data.DataLoader(
            image_datasets[phase],
            batch_size=opt.batchsize,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        for phase in ['train', 'val']
    }
    return dataloaders

# Save Unnormalized Transformed Images
def save_transformed_images(dataloaders):
    os.makedirs(opt.target_folder, exist_ok=True)

    for phase in dataloaders.keys():  # 'train' and 'val'
        phase_folder = os.path.join(opt.target_folder, phase)
        os.makedirs(phase_folder, exist_ok=True)

        dataset = dataloaders[phase].dataset
        transform = dataset.image_transform if phase == 'train' else dataset.transform

        print(f"Processing {phase} dataset...")

        if opt.ours>0:
            our_aug = Ours(probability=opt.ours)
            rpn = RandomErasing(probability = opt.ours, mean=[0.0, 0.0, 0.0])
        # For ImageFolder datasets, the image paths are in dataset.imgs
        for i, (img_path, _) in tqdm(enumerate(dataset.samples), total=len(dataset.samples)):
            img_root = os.path.join(opt.data_dir, 'train' + ('_all' if opt.train_all else ''))
            masks_root = os.path.join(opt.data_dir, 'train_mask' + ('_all' if opt.train_all else ''))
            rel_path = os.path.relpath(img_path, img_root)
            mask_path = os.path.join(masks_root, rel_path)

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")
            
            transformed_img = transform(img)
            transformed_mask = transform(mask)
            
            if opt.ours>0:
                transformed_img,transformed_mask = our_aug(transformed_img, transformed_mask)
                transformed_img = rpn(transformed_img)

            transformed_img = torch.cat([transformed_img, transformed_mask], dim=2)

            # Convert tensor back to PIL image with unnormalization
            if isinstance(transformed_img, torch.Tensor):
                transformed_img = unnormalize(transformed_img)
                transformed_img = transforms.ToPILImage()(transformed_img)

            save_path = os.path.join(phase_folder, f"{i}.jpg")
            transformed_img.save(save_path)

    print(f"All transformed images saved to {opt.target_folder}")

# Main Execution
if __name__ == "__main__":
    dataloaders = load_data()
    save_transformed_images(dataloaders)
