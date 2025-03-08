# Download CLIP from https://github.com/openai/CLIP

import torch
import clip
from torch.utils.data import DataLoader
from moco.loader import NCropsTransform
from utils import get_augmentation, AverageMeter
from image_list import ImageList
from tqdm import tqdm


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def get_augmentation_versions(name, alpha=2.0, kernel_size=3):
    """
    Get a list of augmentations. "w" stands for weak, "s" stands for strong.

    E.g., "wss" stands for one weak, two strong.
    """
    transform_list = [
        get_augmentation("test"),
        get_augmentation(name, alpha=alpha, beta=2.0, patch_height=7, kernel_size=kernel_size),
        get_augmentation(name, alpha=alpha, beta=2.0, patch_height=14, kernel_size=kernel_size),
        get_augmentation(name, alpha=alpha, beta=2.0, patch_height=28, kernel_size=kernel_size),
        get_augmentation(name, alpha=alpha, beta=2.0, patch_height=56, kernel_size=kernel_size),
        get_augmentation(name, alpha=alpha, beta=2.0, patch_height=112, kernel_size=kernel_size),
    ]
    return NCropsTransform(transform_list)

def main(name, alpha, kernel_size=3):
    # Data loading setup
    image_root = 'datasets/PACS'  # Dataset root directory
    label_file = 'datasets/PACS/photo_list.txt'  # Label file
    batch_size = 8
    num_workers = 4

    # Training data
    train_transform = get_augmentation_versions(name, alpha, kernel_size)
    train_dataset = ImageList(
        image_root=image_root,
        label_file=label_file,
        transform=train_transform,
        pseudo_item_list=None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Initialize meters for running averages
    similarity_s1_meter = AverageMeter("Cosine Similarity (S1)", ":.2f")
    similarity_s2_meter = AverageMeter("Cosine Similarity (S2)", ":.2f")
    similarity_s3_meter = AverageMeter("Cosine Similarity (S3)", ":.2f")
    similarity_s4_meter = AverageMeter("Cosine Similarity (S4)", ":.2f")
    similarity_s5_meter = AverageMeter("Cosine Similarity (S5)", ":.2f")

    # Initialize progress meter
    progress = ProgressMeter(
        len(train_loader),
        [similarity_s1_meter, similarity_s2_meter, similarity_s3_meter, similarity_s4_meter, similarity_s5_meter]
    )

    # Process and compare batches of images
    for i, data in enumerate(tqdm(train_loader)):
        # Unpack data
        images, _, _ = data
        images, images_s1, images_s2, images_s3, images_s4, images_s5 = (
            images[0].to(device),
            images[1].to(device),
            images[2].to(device),
            images[3].to(device),
            images[4].to(device),
            images[5].to(device),
        )

        # Compute CLIP embeddings
        with torch.no_grad():
            embeddings = model.encode_image(images)
            embeddings_s1 = model.encode_image(images_s1)
            embeddings_s2 = model.encode_image(images_s2)
            embeddings_s3 = model.encode_image(images_s3)
            embeddings_s4 = model.encode_image(images_s4)
            embeddings_s5 = model.encode_image(images_s5)

        # Compute cosine similarities
        similarity_s1 = torch.nn.functional.cosine_similarity(embeddings, embeddings_s1, dim=-1).mean().item()
        similarity_s2 = torch.nn.functional.cosine_similarity(embeddings, embeddings_s2, dim=-1).mean().item()
        similarity_s3 = torch.nn.functional.cosine_similarity(embeddings, embeddings_s3, dim=-1).mean().item()
        similarity_s4 = torch.nn.functional.cosine_similarity(embeddings, embeddings_s4, dim=-1).mean().item()
        similarity_s5 = torch.nn.functional.cosine_similarity(embeddings, embeddings_s5, dim=-1).mean().item()

        # Update running averages
        similarity_s1_meter.update(similarity_s1, n=batch_size)
        similarity_s2_meter.update(similarity_s2, n=batch_size)
        similarity_s3_meter.update(similarity_s3, n=batch_size)
        similarity_s4_meter.update(similarity_s4, n=batch_size)
        similarity_s5_meter.update(similarity_s5, n=batch_size)

        if i % 50 == 0:
            None # progress.display(i)
    
    progress.display(i)

if __name__ == '__main__':
    # main("moco-v1", 2)

    main("spm_l", 2)
    main("shuffle_patch_mix_l", 2)
    main("shuffle_patch_mix_l", 4)
    main("shuffle_patch_mix_l", 8)
    main("shuffle_patch_mix_l", 16)

    main("spm_o", 2)
    main("shuffle_patch_mix_o", 2)
    main("shuffle_patch_mix_o", 4)
    main("shuffle_patch_mix_o", 8)
    main("shuffle_patch_mix_o", 16)

    main("moco-v2", 2)
    main("jigsaw", 2)

    main("spm", 2)
    main("shuffle_patch_mix", 2)
    main("shuffle_patch_mix", 4)
    main("shuffle_patch_mix", 8)
    main("shuffle_patch_mix", 16)

    main("spm_o_l", 2)
    main("shuffle_patch_mix_o_l", 2)
    main("shuffle_patch_mix_o_l", 4)
    main("shuffle_patch_mix_o_l", 8)
    main("shuffle_patch_mix_o_l", 16)