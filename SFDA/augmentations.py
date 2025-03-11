import torch
import numpy as np
from PIL import Image
import random
import math


def _get_pixels(mode, patch_size, dtype=torch.float32, device='cuda',img_patch=None):
    device=img_patch.device
    if mode == 'rand':
        return (128*torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_())+128
    elif mode == 'pixel':
        return (128*torch.empty(patch_size, dtype=dtype, device=device).normal_())+128
    elif mode == 'soft_pixel':
        return(img_patch + 128*torch.empty(patch_size, dtype=dtype, device=device).normal_())
    else:
        assert not mode or mode == 'const'
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)

class RandomErasing:
    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            mode='const', min_count=1, max_count=None, num_splits=0, device='cuda'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        self.mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                # print(w,h,img_h,img_w) # 130 42 224 224
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.mode, (chan, h, w),
                        dtype=dtype, device=self.device, img_patch=img[:, top:top + h, left:left + w])
                    break

    def __call__(self, input):
        
        input = torch.tensor(np.array(input))
        input = input.to(torch.float32)
        # print(input.shape) # [224, 224, 3]
        
        # Swap dimensions to [3, 224, 224]
        input = input.permute(2, 0, 1)

        if len(input.shape) == 3:
            self._erase(input, *input.shape, input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.shape
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
                
        # Swap dimensions to [3, 224, 224]
        input = input.permute(1, 2, 0)

        # Convert tensor back to NumPy for PIL, ensuring values are in [0, 255]
        input = input.cpu().numpy()  
        input = np.clip(input, 0, 255).astype(np.uint8)
        return Image.fromarray(input)


def peano_curve_indices(num_patches_h, num_patches_w):
    # Simple Peano curve-like ordering
    indices = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            if i % 2 == 0:
                indices.append((i, j))
            else:
                indices.append((i, num_patches_w - 1 - j))
    return np.array(indices).reshape(-1, 2)


class JigsawPuzzle():
    def __init__(self, patch_height, patch_width, mix_prob=0.8):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.mix_prob = mix_prob

    def __call__(self, img):
        if torch.rand(1) > self.mix_prob:
            return img
        img_np = np.array(img)
        h, w, c = img_np.shape
        num_patches_h = h // self.patch_height
        num_patches_w = w // self.patch_width
        N = num_patches_h * num_patches_w

        # Create a list of patches
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h = i * self.patch_height
                start_w = j * self.patch_width
                patch = img_np[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width]
                patches.append(patch)

        # Generate a random permutation of the indices
        indices = np.random.permutation(N)

        # Initialize the mixed patches list
        mixed_patches = []

        # Apply PatchMix transformation
        for i in range(N):
            shuffled_index = indices[i]
            mixed_patch = patches[shuffled_index]
            mixed_patches.append(mixed_patch)

        # Reconstruct the mixed image from mixed patches
        mixed_img = np.zeros_like(img_np)
        index = 0
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h = i * self.patch_height
                start_w = j * self.patch_width
                mixed_img[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width] = mixed_patches[index]
                index += 1

        # Convert back to PIL Image
        mixed_img = np.clip(mixed_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(mixed_img)
    
class JigsawPuzzle_all():
    def __init__(self, patch_height, patch_width, mix_prob=0.8):
        self.patch_height_options = [14, 28, 56, 112]
        self.mix_prob = mix_prob

    def __call__(self, img):
        if  torch.rand(1) > self.mix_prob:
            return img
        img_np = np.array(img)
        h, w, c = img_np.shape

        random_number = np.random.randint(1, len(self.patch_height_options))
        self.patch_height = self.patch_height_options[random_number]
        self.patch_width = self.patch_height_options[random_number] 
        num_patches_h = h // self.patch_height
        num_patches_w = w // self.patch_width
        N = num_patches_h * num_patches_w

        # Create a list of patches
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h = i * self.patch_height
                start_w = j * self.patch_width
                patch = img_np[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width]
                patches.append(patch)

        # Generate a random permutation of the indices
        indices = np.random.permutation(N)

        # Initialize the mixed patches list
        mixed_patches = []

        # Apply PatchMix transformation
        for i in range(N):
            shuffled_index = indices[i]
            mixed_patch = patches[shuffled_index]
            mixed_patches.append(mixed_patch)

        # Reconstruct the mixed image from mixed patches
        mixed_img = np.zeros_like(img_np)
        index = 0
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h = i * self.patch_height
                start_w = j * self.patch_width
                mixed_img[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width] = mixed_patches[index]
                index += 1

        # Convert back to PIL Image
        mixed_img = np.clip(mixed_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(mixed_img)

class JigsawPuzzle_l():
    def __init__(self, patch_height, patch_width, mix_prob=0.8):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.mix_prob = mix_prob

    def __call__(self, img):
        if torch.rand(1) > self.mix_prob:
            return img
        img_np = np.array(img)
        h, w, c = img_np.shape
        num_patches_h = h // self.patch_height
        num_patches_w = w // self.patch_width
        N = num_patches_h * num_patches_w

        # local region of shuffling
        self.M = max(num_patches_h, 4)

        # Create a list of patches
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h = i * self.patch_height
                start_w = j * self.patch_width
                patch = img_np[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width]
                patches.append(patch)

        # Rearrange patches using Peano curve-like scan
        peano_indices = peano_curve_indices(num_patches_h, num_patches_w)
        reordered_patches = [patches[i * num_patches_w + j] for i, j in peano_indices]

        # Shuffle M indices at a time
        mixed_patches = []
        for i in range(0, N, self.M):
            end = min(i + self.M, N)
            group_indices = np.random.permutation(end - i) + i
            for j in range(i, end):
                shuffled_index = group_indices[j - i]
                mixed_patch = reordered_patches[shuffled_index]
                mixed_patches.append(mixed_patch)

        # Reconstruct the mixed image from mixed patches by undoing the Peano scan
        mixed_img = np.zeros_like(img_np)
        index = 0
        for i, (row, col) in enumerate(peano_indices):
            start_h = row * self.patch_height
            start_w = col * self.patch_width
            mixed_img[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width] = mixed_patches[index]
            index += 1

        # Convert back to PIL Image
        mixed_img = np.clip(mixed_img, 0, 255).astype(np.uint8)
        return Image.fromarray(mixed_img)


class RandomPatchNoise():
    def __init__(self, patch_height, patch_width, mix_prob=1.0):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.mix_prob = mix_prob

    def RPN(self, img_patch, rpn_noise=0.25):
        
        if  torch.rand(1) > rpn_noise:
            return(img_patch)
        else:
            return(img_patch + 64*torch.empty((self.patch_height,self.patch_width,3), dtype=img_patch.dtype, device=img_patch.device).normal_())
    
    def __call__(self, img):
        if torch.rand(1) > self.mix_prob:
            return img
        
        img_tensor = torch.tensor(np.array(img))
        img_tensor = img_tensor.to(torch.float32)
        h, w, c = img_tensor.shape
        num_patches_h = h // self.patch_height
        num_patches_w = w // self.patch_width
        N = num_patches_h * num_patches_w

        # Create a list of patches
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h = i * self.patch_height
                start_w = j * self.patch_width
                img_tensor[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width] = self.RPN(img_tensor[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width])


        # Convert tensor back to NumPy for PIL, ensuring values are in [0, 255]
        img_np = img_tensor.cpu().numpy()  
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)
    


    
class RandomPatchErase():
    def __init__(self, patch_height, patch_width, mix_prob=1.0):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.mix_prob = mix_prob

    def RPN(self, img_patch, rpn_noise=0.25):
        
        if  torch.rand(1) > rpn_noise:
            return(img_patch)
        else:
            return(torch.zeros((1, 1, 3), dtype=img_patch.dtype, device=img_patch.device))
    
    def __call__(self, img):
        if torch.rand(1) > self.mix_prob:
            return img
        
        img_tensor = torch.tensor(np.array(img))
        img_tensor = img_tensor.to(torch.float32)
        h, w, c = img_tensor.shape
        num_patches_h = h // self.patch_height
        num_patches_w = w // self.patch_width
        N = num_patches_h * num_patches_w

        # Create a list of patches
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h = i * self.patch_height
                start_w = j * self.patch_width
                img_tensor[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width] = self.RPN(img_tensor[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width])


        # Convert tensor back to NumPy for PIL, ensuring values are in [0, 255]
        img_np = img_tensor.cpu().numpy()  
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)