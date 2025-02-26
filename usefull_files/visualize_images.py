import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.utils import make_grid
from moco.loader import NCropsTransform
from utils import get_augmentation
from image_list import ImageList

def unnormalize(tensor, mean, std):
    """Unnormalize a tensor image."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def create_montage(images, mean, std):
    """Create a montage from a batch of images."""
    images = [unnormalize(img.cpu(), mean, std) for img in images]
    grid = make_grid(images, nrow=4)  # Create a 4x4 grid
    return grid 

def save_montage(images_w, images_q, images_k, filename='montage.png'):
    """Save three sets of images in a 1x3 grid as an image."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    montage_w = create_montage(images_w, mean, std)
    montage_q = create_montage(images_q, mean, std)
    montage_k = create_montage(images_k, mean, std)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(transforms.ToPILImage()(montage_w))
    axes[0].set_title('Weak Augmentation')
    axes[0].axis("off")
    
    axes[1].imshow(transforms.ToPILImage()(montage_q))
    axes[1].set_title('Strong Augmentation 1')
    axes[1].axis("off")
    
    axes[2].imshow(transforms.ToPILImage()(montage_k))
    axes[2].set_title('Strong Augmentation 2')
    axes[2].axis("off")
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

# Assuming this function exists and is correctly implemented
def get_augmentation_versions():
    transform_list = [
        get_augmentation("test"),
        get_augmentation("mask"), 
        get_augmentation("shuffle_patch_mix"), 
    ]
    return NCropsTransform(transform_list)

def main():
    # Example usage with the specified replacements
    image_root = 'datasets/PACS'  # Replace args.data.image_root
    pseudo_item_list = None  # Replace pseudo_item_list
    batch_size = 16  # Replace args.data.batch_size
    num_workers = 1  # Replace args.data.workers
    label_file = 'datasets/PACS/photo_list.txt'

    # Training data
    train_transform = get_augmentation_versions()
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
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False,
    )

    # Save images during the training loop
    for i, data in enumerate(train_loader):
        # unpack and move data
        images, _, idxs = data
        idxs = idxs.to("cuda")
        images_w, images_q, images_k = (
            images[0].to("cuda"),
            images[1].to("cuda"),
            images[2].to("cuda"),
        )

        # Save the three sets of images in a montage
        save_montage(images_w, images_q, images_k, filename=f'output/visualize/montage_batch_{i}.png')
        
        # Break after saving the first five batchs
        if i==10:
            break

if __name__ == '__main__':
    main()
