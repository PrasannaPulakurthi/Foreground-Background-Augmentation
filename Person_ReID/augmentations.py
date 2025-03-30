from __future__ import absolute_import
import random
import math
import torch
import numpy as np


class Ours(object):
    def __init__(self, probability=0.5):
        self.probability = probability
        self.jigsaw = JigsawPuzzle(mix_prob=probability, shuffle_ratio=0.5)
        self.rpn = RandomPatchNoise(probability=probability, noise_strength=3.0)

    def process_single(self, img, mask):
        # img and mask are assumed to be tensors of shape [C,H,W]
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])  # Horizontal flip: flip width dim
            mask = torch.flip(mask, dims=[2])
        
        if random.uniform(0, 1) > (self.probability):
            return  img, mask
        
        mask = mask.mean(dim=0, keepdim=True).repeat(3, 1, 1)  # Convert to grayscale shape: [3, H, W]

        mask = (normPRED(mask) > 0.5).float()

        # If the mask is almost empty, apply random patch noise
        mask_val = 0.5
        if mask.mean() < mask_val:
            mm = mask.mean()/mask_val
            img_out = ((img * (1-mm)) + (self.jigsaw(img) * mm))
            img_out = (self.rpn(img) * mask + img_out * (1 - mask))
            return img_out, mask

        # Apply jigsaw to img and combine using the mask
        img_bg = self.jigsaw(img) * (1 - mask)
        img_fg = self.rpn(img) * mask 
        img_out = img_fg + img_bg
        return img_out, mask

    def __call__(self, img, mask):
        # If input is a single sample [C,H,W]
        if img.dim() == 3:
            return self.process_single(img, mask)
        # If input is batched [N,C,H,W], process each sample individually
        elif img.dim() == 4:
            out_imgs = []
            out_masks = []
            for i in range(img.size(0)):
                proc_img, proc_mask = self.process_single(img[i], mask[i])
                out_imgs.append(proc_img)
                out_masks.append(proc_mask)
            return torch.stack(out_imgs), torch.stack(out_masks)
        else:
            raise ValueError("Input image must have 3 or 4 dimensions.")


class JigsawPuzzle():
    def __init__(self, patch_height=64, patch_width=64, mix_prob=0.8, shuffle_ratio=0.5):
        # You can also keep the options if needed
        self.patch_height_options = [16, 32, 64, 128]
        self.patch_width_options = [16, 32, 64]
        self.mix_prob = mix_prob
        self.shuffle_ratio = shuffle_ratio  # Percentage of patches to shuffle

    def process_single(self, img):
        # img is a single image tensor with shape [C,H,W]
        if torch.rand(1).item() > self.mix_prob:
            return img

        # Convert tensor to numpy array with shape (H, W, C)
        img_np = img.permute(1, 2, 0).cpu().numpy()
        h, w, c = img_np.shape

        # Randomly select patch size from options (skipping index 0 to vary sizes)
        random_number_h = np.random.randint(1, len(self.patch_height_options))
        patch_height = self.patch_height_options[random_number_h]
        random_number_w = np.random.randint(1, len(self.patch_width_options))
        patch_width = self.patch_width_options[random_number_w]

        num_patches_h = h // patch_height
        num_patches_w = w // patch_width

        # Reshape into grid of patches: (num_patches_h, patch_height, num_patches_w, patch_width, c)
        patches = img_np.reshape(num_patches_h, patch_height, num_patches_w, patch_width, c)
        # Rearrange to (num_patches, patch_height, patch_width, c)
        patches = patches.swapaxes(1, 2).reshape(-1, patch_height, patch_width, c)

        num_patches = patches.shape[0]
        num_to_shuffle = int(num_patches * self.shuffle_ratio)
        if num_to_shuffle > 0:
            shuffle_indices = np.random.choice(num_patches, num_to_shuffle, replace=False)
            selected_patches = patches[shuffle_indices].copy()
            np.random.shuffle(selected_patches)
            patches[shuffle_indices] = selected_patches

        # Reshape back to original grid structure
        mixed_img = patches.reshape(num_patches_h, num_patches_w, patch_height, patch_width, c)
        mixed_img = mixed_img.swapaxes(1, 2).reshape(h, w, c)

        # Convert back to tensor: shape [C,H,W]
        mixed_img_tensor = torch.from_numpy(mixed_img).to(img.device).type_as(img).permute(2, 0, 1)
        return mixed_img_tensor

    def __call__(self, img):
        # If a single image [C,H,W]
        if img.dim() == 3:
            return self.process_single(img)
        # If batched [N,C,H,W], process each sample individually
        elif img.dim() == 4:
            out_imgs = []
            for i in range(img.size(0)):
                out_imgs.append(self.process_single(img[i]))
            return torch.stack(out_imgs)
        else:
            raise ValueError("Input image must have 3 or 4 dimensions.")

class RandomPatchNoise():
    def __init__(self, patch_height=64, patch_width=64, probability=1.0, noise_strength=5.0):
        self.patch_height_options = [16, 32, 64, 128]
        self.patch_width_options = [16, 32, 64]
        self.probability = probability
        self.sl = 0.02
        self.sh = 0.4
        self.noise_strength = noise_strength

    def _apply_patch_noise(self, img_tensor):
        """Apply random patch noise to a single image tensor of shape [C, H, W]."""
        if torch.rand(1) > self.probability:
            return img_tensor

        # Randomly select patch size from the options.
        random_number_h = np.random.randint(1, len(self.patch_height_options))
        patch_height = self.patch_height_options[random_number_h]
        random_number_w = np.random.randint(1, len(self.patch_width_options))
        patch_width = self.patch_width_options[random_number_w]

        c, h, w = img_tensor.shape  # e.g., (3, 256, 128)
        num_patches_h = h // patch_height
        num_patches_w = w // patch_width

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h, end_h, start_w, end_w = find_index(i, j, patch_height, patch_width, h, w)
                rpn_prob = torch.distributions.uniform.Uniform(self.sl, self.sh).sample([1])
                if torch.rand(1) < rpn_prob:
                    img_patch = img_tensor[:, start_h:end_h, start_w:end_w]
                    noisy_patch = img_patch + self.noise_strength * torch.randn_like(img_patch)
                    img_tensor[:, start_h:end_h, start_w:end_w] = noisy_patch

        return img_tensor

    def __call__(self, img):
        """
        Apply random patch noise on a single image ([C, H, W]) or a minibatch ([N, C, H, W]).
        """
        if img.dim() == 3:  # Single image
            return self._apply_patch_noise(img)
        elif img.dim() == 4:  # Minibatch of images
            for i in range(img.size(0)):
                img[i] = self._apply_patch_noise(img[i])
            return img
        else:
            raise ValueError("Input must be a 3D tensor (C, H, W) or a 4D tensor (N, C, H, W)")
        
'''
class RandomPatchNoise(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0, 0, 0], noise_std=3.0):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.noise_std = noise_std
       
    def _erase_image(self, img):
        """Apply random erasing to a single image tensor of shape [C, H, W]."""
        if random.uniform(0, 1) > self.probability:
            return img

        # Try up to 100 times to find a valid erasing area.
        for attempt in range(100):
            # Compute area of the image.
            area = img.size(1) * img.size(2)
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            # Check if the rectangle fits in the image dimensions.
            if w < img.size(2) and h < img.size(1):
                x1 = random.randint(0, img.size(1) - h)
                y1 = random.randint(0, img.size(2) - w)
                if img.size(0) == 3:  # for 3-channel images
                    # Generate noise for each channel; each channel's noise is centered on its mean.
                    noise = torch.randn((3, h, w)) * self.noise_std + torch.tensor(self.mean).view(3, 1, 1)
                    img[:, x1:x1+h, y1:y1+w] = img[:, x1:x1+h, y1:y1+w] + noise
                else:  # for single-channel images
                    noise = torch.randn((1, h, w)) * self.noise_std + torch.tensor([self.mean[0]]).view(1, 1, 1)
                    img[0, x1:x1+h, y1:y1+w] = img[0, x1:x1+h, y1:y1+w] + noise
                return img

        # Return the image unchanged if no valid area was found.
        return img

    def __call__(self, img):
        """
        Apply random erasing on either a single image (shape: [C, H, W])
        or a minibatch of images (shape: [N, C, H, W]).
        """
        if img.dim() == 3:  # Single image
            return self._erase_image(img)
        elif img.dim() == 4:  # Minibatch of images
            # Iterate over each image in the batch
            for i in range(img.size(0)):
                img[i] = self._erase_image(img[i])
            return img
        else:
            raise ValueError("Unsupported tensor dimension. Expected 3D or 4D tensor.")

'''


class RandomErasing(object):
    """Randomly selects a rectangle region in an image and erases its pixels.
       'Random Erasing Data Augmentation' by Zhong et al.
       See https://arxiv.org/pdf/1708.04896.pdf

    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def _erase_image(self, img):
        """Apply random erasing to a single image tensor of shape [C, H, W]."""
        if random.uniform(0, 1) > self.probability:
            return img

        # Try up to 100 times to find a valid erasing area.
        for attempt in range(100):
            # Compute area of the image.
            area = img.size(1) * img.size(2)
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            # Check if the rectangle fits in the image dimensions.
            if w < img.size(2) and h < img.size(1):
                x1 = random.randint(0, img.size(1) - h)
                y1 = random.randint(0, img.size(2) - w)
                if img.size(0) == 3:  # for 3-channel images
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:  # for single-channel images
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        # Return the image unchanged if no valid area was found.
        return img

    def __call__(self, img):
        """
        Apply random erasing on either a single image (shape: [C, H, W])
        or a minibatch of images (shape: [N, C, H, W]).
        """
        if img.dim() == 3:  # Single image
            return self._erase_image(img)
        elif img.dim() == 4:  # Minibatch of images
            # Iterate over each image in the batch
            for i in range(img.size(0)):
                img[i] = self._erase_image(img[i])
            return img
        else:
            raise ValueError("Unsupported tensor dimension. Expected 3D or 4D tensor.")


class RandomGrayscaleErasing(object):
    """ Randomly selects a rectangle region in an image and use grayscale image
        instead of its pixels.
        'Local Grayscale Transfomation' by Yunpeng Gong.
        See https://arxiv.org/pdf/2101.08533.pdf
    Args:
         probability: The probability that the Random Grayscale Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
    """

    def __init__(self, probability: float = 0.2, sl: float = 0.02, sh: float = 0.4, r1: float = 0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        """
        Args:
            img: after ToTensor() and Normalize([...]), img's type is Tensor
        """
        if random.uniform(0, 1) > self.probability:
            return img

        height, width = img.size()[-2], img.size()[-1]
        area = height * width

        for _ in range(100):

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)  # height / width

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                # tl
                x = random.randint(0, height - h)
                y = random.randint(0, width - w)
                # unbind channel dim
                r, g, b = img.unbind(dim=-3)
                # Weighted average method -> grayscale patch
                l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
                l_img = l_img.unsqueeze(dim=-3)  # rebind channel
                # erasing
                img[0, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]
                img[1, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]
                img[2, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]

                return img

        return img


def normalize(tensor):
    # Define the ImageNet mean and std for each channel
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device) 
    # Normalize the tensor: (tensor - mean) / std
    tensor_normalized = (tensor - mean) / std
    return tensor_normalized

# Unnormalization transformation
def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean  # Reverse normalization
    tensor = torch.clamp(tensor, 0, 1)  # Clamp to valid image range
    tensor = tensor/torch.max(tensor) # Range [0,1]
    return tensor

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def find_index(i,j,patch_height,patch_width,h,w):
    # Base grid patch starting coordinates.
    start_h = i * patch_height
    start_w = j * patch_width

    # Define maximum shift (e.g., 25% of patch size)
    shift_range_h = int(patch_height * 0.25)
    shift_range_w = int(patch_width * 0.25)
    offset_h = np.random.randint(-shift_range_h, shift_range_h + 1)
    offset_w = np.random.randint(-shift_range_w, shift_range_w + 1)

    # Apply the random offset.
    new_start_h = start_h + offset_h
    new_start_w = start_w + offset_w

    # Ensure the new patch is completely within the image.
    new_start_h = max(0, min(new_start_h, h - patch_height))
    new_start_w = max(0, min(new_start_w, w - patch_width))
    new_end_h = new_start_h + patch_height
    new_end_w = new_start_w + patch_width

    return new_start_h,new_end_h,new_start_w,new_end_w

