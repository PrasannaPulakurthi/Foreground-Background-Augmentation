import logging
import math
import os
from PIL import Image
import yaml

from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

from moco.loader import GaussianBlur
import numpy as np
from augmentations import JigsawPuzzle, JigsawPuzzle_l, JigsawPuzzle_all, RandomErasing, RandomPatchNoise, RandomPatchErase


LOG_FORMAT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

NUM_CLASSES = {"domainnet-126": 126, "VISDA-C": 12, "PACS": 7}

import torch
import numpy as np
from PIL import Image


def configure_logger(rank, log_path=None):
    if log_path:
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)

    # only master process will print & write
    level = logging.INFO if rank in {-1, 0} else logging.WARNING
    handlers = [logging.StreamHandler()]
    if rank in {0, -1} and log_path:
        handlers.append(logging.FileHandler(log_path, "w"))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=handlers,
    )


class UnevenBatchLoader:
    """Loader that loads data from multiple datasets with different length."""

    def __init__(self, data_loaders, is_ddp=False):
        # register N data loaders with epoch counters.
        self.data_loaders = data_loaders
        self.epoch_counters = [0 for _ in range(len(data_loaders))]

        # set_epoch() needs to be called before creating the iterator
        self.is_ddp = is_ddp
        if is_ddp:
            for data_loader in data_loaders:
                data_loader.sampler.set_epoch(0)
        self.iterators = [iter(data_loader) for data_loader in data_loaders]

    def next_batch(self):
        """Load the next batch by collecting from N data loaders.
        Args:
            None
        Returns:
            data: a list of N items from N data loaders. each item has the format
                output by a single data loader.
        """
        data = []
        for i, iterator in enumerate(self.iterators):
            try:
                batch_i = next(iterator)
            except StopIteration:
                self.epoch_counters[i] += 1
                # create a new iterator
                if self.is_ddp:
                    self.data_loaders[i].sampler.set_epoch(self.epoch_counters[i])
                new_iterator = iter(self.data_loaders[i])
                self.iterators[i] = new_iterator
                batch_i = next(new_iterator)
            data.append(batch_i)

        return data

    def update_loader(self, idx, loader, epoch=None):
        if self.is_ddp and isinstance(epoch, int):
            loader.sampler.set_epoch(epoch)
        self.iterators[idx] = iter(loader)


class CustomDistributedDataParallel(DistributedDataParallel):
    """A wrapper class over DDP that relay "module" attribute."""

    def __init__(self, model, **kwargs):
        super(CustomDistributedDataParallel, self).__init__(model, **kwargs)

    def __getattr__(self, name):
        try:
            return super(CustomDistributedDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def remove_wrap_arounds(tensor, ranks):
    if ranks == 0:
        return tensor

    world_size = dist.get_world_size()
    single_length = len(tensor) // world_size
    output = []
    for rank in range(world_size):
        sub_tensor = tensor[rank * single_length : (rank + 1) * single_length]
        if rank >= ranks:
            output.append(sub_tensor[:-1])
        else:
            output.append(sub_tensor)
    output = torch.cat(output)

    return output


def get_categories(category_file):
    """Return a list of categories ordered by corresponding label.

    Args:
        category_file: str, path to the category file. can be .yaml or .txt

    Returns:
        categories: List[str], a list of categories ordered by label.
    """
    if category_file.endswith(".yaml"):
        with open(category_file, "r") as fd:
            cat_mapping = yaml.load(fd, Loader=yaml.SafeLoader)
        categories = list(cat_mapping.keys())
        categories.sort(key=lambda x: cat_mapping[x])
    elif category_file.endswith(".txt"):
        with open(category_file, "r") as fd:
            categories = fd.readlines()
        categories = [cat.strip() for cat in categories if cat]
    else:
        raise NotImplementedError()

    categories = [cat.replace("_", " ") for cat in categories]
    return categories


def get_augmentation(aug_type, patch_height=28, mix_prob=0.8, normalize=None):
    if not normalize:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    if aug_type == "moco-v2":
        image_aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "moco-v1":
        image_aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "plain":
        image_aug = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "clip_inference":
        image_aug = transforms.Compose(
            [
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "test":
        image_aug = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "jigsaw":
        image_aug =  transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                # transforms.RandomHorizontalFlip(),
                JigsawPuzzle(patch_height=patch_height, patch_width=patch_height, mix_prob=1),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "jigsaw_all":
        image_aug =  transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                # transforms.RandomHorizontalFlip(),
                JigsawPuzzle_all(patch_height=patch_height, patch_width=patch_height, mix_prob=1),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "jigsaw_l":
        image_aug = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                # transforms.RandomHorizontalFlip(),
                JigsawPuzzle_l(patch_height=patch_height, patch_width=patch_height, mix_prob=1),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "rpe":
        image_aug = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                # transforms.RandomHorizontalFlip(),
                RandomPatchErase(patch_height=patch_height, patch_width=patch_height, mix_prob=1),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "rpn":
        image_aug = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                # transforms.RandomHorizontalFlip(),
                RandomPatchNoise(patch_height=patch_height, patch_width=patch_height, mix_prob=1),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type in ["ours", "ours_1"]:
        image_aug = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                JigsawPuzzle_all(patch_height=patch_height, patch_width=patch_height, mix_prob=mix_prob),
                transforms.ToTensor(),
            ]
        )
    else:
        image_aug = None

    return DualTransform(
        aug_type=aug_type,
        image_transform=image_aug,
        patch_height=patch_height, 
        patch_width=patch_height,
        mix_prob=mix_prob,
    )

def fuse_foreground_background(img1, img2, mask):
    """
    Given a (C,H,W) image tensor and a (possibly 2D) mask,
    multiply img by mask to black out the background.
    Expects 0 as background in the mask.
    """
    mask = (mask > 0.5)
    output = img1 * mask + img2 * (~mask)

    return output

def normalize(tensor):
    T = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    return T(tensor)

class DualTransform:
    """
    A wrapper that can apply image-only transforms or image+mask transforms.
    """
    def __init__(self, aug_type, image_transform=None, patch_height=28, patch_width=28,mix_prob=1.0):
        self.image_transform = image_transform
        self.aug_type = aug_type
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        self.moco_transform = transforms.Compose(
            [   
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                # RandomErasing(mode='soft_pixel'),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.fpn = RandomPatchNoise(patch_height=28, patch_width=28, mix_prob=mix_prob)
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.jigsaw = JigsawPuzzle(patch_height=28, patch_width=28, mix_prob=mix_prob)
        self.jigsaw_all = JigsawPuzzle_all(mix_prob=mix_prob)

    def __call__(self, img, mask=None):
        if self.aug_type == "mask":
            mask = self.base_transform(mask)
            return normalize(mask)
        elif self.aug_type == "foreground":
            mask = self.base_transform(mask)
            img = self.base_transform(img)
            return normalize(img * (mask>0.5).float())
        elif self.aug_type == "fpn":
            mask = self.base_transform(mask)
            img = self.base_transform(img)
            img_n = self.to_tensor(self.fpn(self.to_pil(img)))
            return normalize(img_n * (mask>0.5).float())
        elif self.aug_type == "bps":
            mask = self.base_transform(mask)
            img = self.base_transform(img)
            img_jigsaw = self.to_tensor(self.jigsaw_all(self.to_pil(img)))
            return normalize(img_jigsaw * (mask<0.5).float())
        elif self.aug_type == "ours_raw":
            mask = self.base_transform(mask)
            img = self.base_transform(img)
            img_n = self.to_tensor(self.fpn(self.to_pil(img)))
            img_jigsaw = self.to_tensor(self.jigsaw_all(self.to_pil(img)))
            img_out = fuse_foreground_background(img_n, img_jigsaw, mask)
            return normalize(img_out)
        elif self.aug_type == "ours":
            mask = self.base_transform(mask)
            img = self.base_transform(img)
            img_n = self.to_tensor(self.fpn(self.to_pil(img)))
            img_jigsaw = self.to_tensor(self.jigsaw_all(self.to_pil(img)))
            img_out = fuse_foreground_background(img_n, img_jigsaw, mask)
            return self.moco_transform(self.to_pil(img_out))
        elif self.aug_type == "ours_fpn":
            mask = self.base_transform(mask)
            img = self.base_transform(img)
            img_n = self.to_tensor(self.fpn(self.to_pil(img)))
            img_out = fuse_foreground_background(img_n, img, mask)
            return self.moco_transform(self.to_pil(img_out))
        elif self.aug_type == "ours_bps":
            mask = self.base_transform(mask)
            img = self.base_transform(img)
            # img_n = self.to_tensor(self.fpn(self.to_pil(img)))
            img_jigsaw = self.to_tensor(self.jigsaw_all(self.to_pil(img)))
            img_out = fuse_foreground_background(img, img_jigsaw, mask)
            return self.moco_transform(self.to_pil(img_out))
        # Always transform the image if we have an image_transform
        else:
            return self.image_transform(img)
        '''
        elif self.aug_type == "ours_old":
            img_t = self.image_transform(img)
            img = self.base_transform(img)
            mask = self.base_transform(mask)
            img_t1 = fuse_foreground_background(img, img_t, mask)
            img_t1_pil = self.to_pil(img_t1)
            output = self.moco_transform(img_t1_pil)
            return output
        elif self.aug_type == "ours_1":
            img_t = self.image_transform(img)
            img = self.base_transform(img)
            mask = self.base_transform(mask)
            img_t1 = fuse_foreground_background(img, img_t, mask)
            return normalize(img_t1)'
        '''
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def save_checkpoint(model, optimizer, epoch, save_path="checkpoint.pth.tar"):
    state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, save_path)


def adjust_learning_rate(optimizer, progress, args):
    """
    Decay the learning rate based on epoch or iteration.
    """
    if args.optim.cos:
        decay = 0.5 * (1.0 + math.cos(math.pi * progress / args.learn.full_progress))
    elif args.optim.exp:
        decay = (1 + 10 * progress / args.learn.full_progress) ** -0.75
    else:
        decay = 1.0
        for milestone in args.optim.schedule:
            decay *= args.optim.gamma if progress >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay

    return decay


def per_class_accuracy(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    acc_per_class = (matrix.diagonal() / matrix.sum(axis=1) * 100.0).round(2)
    logging.info(
        f"Accuracy per class: {acc_per_class}, mean: {acc_per_class.mean().round(2)}"
    )

    return acc_per_class


def get_distances(X, Y, dist_type="euclidean"):
    """
    Args:
        X: (N, D) tensor
        Y: (M, D) tensor
    """
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances


def is_master(args):
    return args.rank % args.ngpus_per_node == 0


def use_wandb(args):
    return is_master(args) and args.use_wandb
