import torch
import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

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

def create_feather_mask(height, width, feather_size=4):
    """
    Create a 2D mask of shape (height x width) that smoothly transitions
    from 1.0 in the interior to 0.0 at the edges over 'feather_size' pixels.
    """
    mask = np.ones((height, width), dtype=np.float32)
    ramp = np.linspace(0, 1, feather_size, dtype=np.float32)

    # Top fade
    mask[:feather_size, :]    *= ramp[:, None]
    # Bottom fade
    mask[-feather_size:, :]   *= ramp[::-1, None]
    # Left fade
    mask[:, :feather_size]    *= ramp[None, :]
    # Right fade
    mask[:, -feather_size:]   *= ramp[None, ::-1]

    return mask

def edgelogic(i, j, patch_height, patch_width, num_patches_h, num_patches_w, overlap):
    """
    Example 'edgelogic' that extends patch size in the middle,
    but does not exceed (patch_height+2*overlap, patch_width+2*overlap).
    Modify as needed for your scenario.
    """
    # Base top-left (no overlap):
    start_h = i * patch_height
    start_w = j * patch_width
    end_h   = start_h + patch_height
    end_w   = start_w + patch_width

    # If i == 0, we add overlap only at the bottom. If i == last, only top, etc.
    # This is just one possible logic:
    if i == 0:
        end_h += 2 * overlap
    elif i == num_patches_h - 1:
        start_h -= 2 * overlap
    else:
        start_h -= overlap
        end_h   += overlap

    if j == 0:
        end_w += 2 * overlap
    elif j == num_patches_w - 1:
        start_w -= 2 * overlap
    else:
        start_w -= overlap
        end_w   += overlap

    # Make sure we don't go negative or beyond image dimension here if needed

    return start_h, end_h, start_w, end_w


class SAM2():
    def __init__(self):
        checkpoint = "sam2/checkpoints/sam2.1_hiera_small.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        self.sam2 = build_sam2(model_cfg, checkpoint, apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2)   

    def __call__(self, img):
        img_np = np.array(img)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = self.mask_generator.generate(img_np)
        print(len(masks))
        print(masks[0].keys())
        return(masks)


class ShufflePatchMix():
    def __init__(self, patch_height, patch_width, mix_prob=0.8, alpha=4.0, beta=2.0):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta

    def __call__(self, img):
        if  torch.rand(1) > self.mix_prob:
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
            # Mixing using Lambda sampled from a Beta distribution
            lambda_ = np.random.beta(self.alpha, self.beta)
            shuffled_index = indices[i]
            mixed_patch = lambda_ * patches[i] + (1 - lambda_) * patches[shuffled_index]
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

class ShufflePatchMix_all():
    def __init__(self, patch_height, patch_width, mix_prob=0.8, alpha=4.0, beta=2.0):
        patch_height_options = [14, 28, 56, 112]
        random_number = np.random.randint(1, len(patch_height_options))
        self.patch_height = patch_height_options[random_number]
        self.patch_width = patch_height_options[random_number] 
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta

    def __call__(self, img):
        if  torch.rand(1) > self.mix_prob:
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
            # Mixing using Lambda sampled from a Beta distribution
            lambda_ = np.random.beta(self.alpha, self.beta)
            shuffled_index = indices[i]
            mixed_patch = lambda_ * patches[i] + (1 - lambda_) * patches[shuffled_index]
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

class ShufflePatchMix_l():
    def __init__(self, patch_height, patch_width, mix_prob=0.8, alpha=4.0, beta=2.0):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta

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
                lambda_ = np.random.beta(self.alpha, self.beta)
                mixed_patch = lambda_ * reordered_patches[j] + (1 - lambda_) * reordered_patches[shuffled_index]
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

class ShufflePatchMix_l_all():
    def __init__(self, patch_height, patch_width, mix_prob=0.8, alpha=4.0, beta=2.0):
        patch_height_options = [14, 28, 56, 112]
        random_number = np.random.randint(1, len(patch_height_options))
        self.patch_height = patch_height_options[random_number]
        self.patch_width = patch_height_options[random_number] 
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta

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
                lambda_ = np.random.beta(self.alpha, self.beta)
                mixed_patch = lambda_ * reordered_patches[j] + (1 - lambda_) * reordered_patches[shuffled_index]
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

class ShufflePatchMixOverlap():
    def __init__(self, 
                 patch_height, 
                 patch_width, 
                 mix_prob=0.8, 
                 alpha=4.0, 
                 beta=2.0, 
                 overlap=4):
        """
        patch_height, patch_width : Base size of each patch (without overlap).
        overlap                  : Overlap in pixels on each edge.
        mix_prob                 : Probability of applying ShufflePatchMix.
        alpha, beta              : Beta distribution parameters (lambda sampling).
        """
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta
        self.overlap = int(self.patch_height/7)

    def __call__(self, img):
        # 1) Randomly skip
        if torch.rand(1) > self.mix_prob:
            return img

        # 2) Convert to NumPy float32
        img_np = np.array(img, dtype=np.float32)
        h, w, c = img_np.shape

        # 3) Compute # of patches in each dimension
        num_patches_h = h // self.patch_height
        num_patches_w = w // self.patch_width

        # 4) Precompute a large feather mask for the largest possible patch 
        #    (patch_height + 2*overlap x patch_width + 2*overlap)
        #    We'll slice it down for boundary patches if needed
        feather_mask_full = create_feather_mask(
            self.patch_height + 2*self.overlap,
            self.patch_width  + 2*self.overlap,
            feather_size=4
        )

        # 5) Extract patches
        patches = []
        coords = []
        for i_patch in range(num_patches_h):
            for j_patch in range(num_patches_w):
                start_h, end_h, start_w, end_w = edgelogic(
                    i_patch, j_patch,
                    self.patch_height, self.patch_width,
                    num_patches_h, num_patches_w,
                    self.overlap
                )
                # Clip to image boundaries if needed
                start_h = max(0, start_h)
                start_w = max(0, start_w)
                end_h   = min(h, end_h)
                end_w   = min(w, end_w)

                patch = img_np[start_h:end_h, start_w:end_w]
                patches.append(patch)
                coords.append((start_h, end_h, start_w, end_w))

        # 6) Shuffle & mix patches
        N = len(patches)
        indices = np.random.permutation(N)
        mixed_patches = []
        for i_patch in range(N):
            lam = np.random.beta(self.alpha, self.beta)
            patchA = patches[i_patch]
            patchB = patches[indices[i_patch]]
            mixed_patch = lam * patchA + (1 - lam) * patchB
            mixed_patches.append(mixed_patch)

        # 7) Prepare output & weight arrays for soft blending
        output = np.zeros_like(img_np, dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)

        # 8) Blend each patch with the feather mask
        for i_patch, (sh, eh, sw, ew) in enumerate(coords):
            patch_mixed = mixed_patches[i_patch]
            ph, pw, _ = patch_mixed.shape

            # Extract the corresponding portion of the big feather mask
            # if patch is smaller near boundaries, slice the mask
            mask_2d = feather_mask_full[:ph, :pw]
            
            # Convert mask to 3 channels if needed
            if c == 1:
                mask_3d = mask_2d[..., None]
            else:
                mask_3d = np.repeat(mask_2d[..., None], c, axis=2)

            # Feathered patch
            patch_feathered = patch_mixed * mask_3d

            # Accumulate in output
            output[sh:eh, sw:ew] += patch_feathered
            weight[sh:eh, sw:ew] += mask_2d

        # 9) Final divide by weight -> smooth blend
        weight = np.clip(weight, 1e-8, None)
        output /= weight[..., None]

        final_output = img_np
        final_output[1:-1,1:-1] = output[1:-1,1:-1]
        # 10) Convert back to uint8
        final_output = np.clip(final_output, 0, 255).astype(np.uint8)
        return Image.fromarray(final_output)

class ShufflePatchMixOverlap_all():
    def __init__(self, 
                 patch_height, 
                 patch_width, 
                 mix_prob=0.8, 
                 alpha=4.0, 
                 beta=2.0):
        """
        patch_height, patch_width : Base size of each patch (without overlap).
        overlap                  : Overlap in pixels on each edge.
        mix_prob                 : Probability of applying ShufflePatchMix.
        alpha, beta              : Beta distribution parameters (lambda sampling).
        """
        patch_height_options = [14, 28, 56, 112]
        random_number = np.random.randint(1, len(patch_height_options))
        self.patch_height = patch_height_options[random_number]
        self.patch_width = patch_height_options[random_number] 
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta
        self.overlap = int(self.patch_height/7)

    def __call__(self, img):
        # 1) Randomly skip
        if torch.rand(1) > self.mix_prob:
            return img

        # 2) Convert to NumPy float32
        img_np = np.array(img, dtype=np.float32)
        h, w, c = img_np.shape

        # 3) Compute # of patches in each dimension
        num_patches_h = h // self.patch_height
        num_patches_w = w // self.patch_width

        # 4) Precompute a large feather mask for the largest possible patch 
        #    (patch_height + 2*overlap x patch_width + 2*overlap)
        #    We'll slice it down for boundary patches if needed
        feather_mask_full = create_feather_mask(
            self.patch_height + 2*self.overlap,
            self.patch_width  + 2*self.overlap,
            feather_size=4
        )

        # 5) Extract patches
        patches = []
        coords = []
        for i_patch in range(num_patches_h):
            for j_patch in range(num_patches_w):
                start_h, end_h, start_w, end_w = edgelogic(
                    i_patch, j_patch,
                    self.patch_height, self.patch_width,
                    num_patches_h, num_patches_w,
                    self.overlap
                )
                # Clip to image boundaries if needed
                start_h = max(0, start_h)
                start_w = max(0, start_w)
                end_h   = min(h, end_h)
                end_w   = min(w, end_w)

                patch = img_np[start_h:end_h, start_w:end_w]
                patches.append(patch)
                coords.append((start_h, end_h, start_w, end_w))

        # 6) Shuffle & mix patches
        N = len(patches)
        indices = np.random.permutation(N)
        mixed_patches = []
        for i_patch in range(N):
            lam = np.random.beta(self.alpha, self.beta)
            patchA = patches[i_patch]
            patchB = patches[indices[i_patch]]
            mixed_patch = lam * patchA + (1 - lam) * patchB
            mixed_patches.append(mixed_patch)

        # 7) Prepare output & weight arrays for soft blending
        output = np.zeros_like(img_np, dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)

        # 8) Blend each patch with the feather mask
        for i_patch, (sh, eh, sw, ew) in enumerate(coords):
            patch_mixed = mixed_patches[i_patch]
            ph, pw, _ = patch_mixed.shape

            # Extract the corresponding portion of the big feather mask
            # if patch is smaller near boundaries, slice the mask
            mask_2d = feather_mask_full[:ph, :pw]
            
            # Convert mask to 3 channels if needed
            if c == 1:
                mask_3d = mask_2d[..., None]
            else:
                mask_3d = np.repeat(mask_2d[..., None], c, axis=2)

            # Feathered patch
            patch_feathered = patch_mixed * mask_3d

            # Accumulate in output
            output[sh:eh, sw:ew] += patch_feathered
            weight[sh:eh, sw:ew] += mask_2d

        # 9) Final divide by weight -> smooth blend
        weight = np.clip(weight, 1e-8, None)
        output /= weight[..., None]

        final_output = img_np
        final_output[1:-1,1:-1] = output[1:-1,1:-1]
        # 10) Convert back to uint8
        final_output = np.clip(final_output, 0, 255).astype(np.uint8)
        return Image.fromarray(final_output)

class ShufflePatchMixOverlap_l():
    def __init__(self, 
                 patch_height, 
                 patch_width, 
                 mix_prob=0.8, 
                 alpha=4.0, 
                 beta=2.0, 
                 overlap=4):
        """
        patch_height, patch_width : Base size of each patch (without overlap).
        overlap                  : Overlap in pixels on each edge.
        mix_prob                 : Probability of applying ShufflePatchMix.
        alpha, beta              : Beta distribution parameters (lambda sampling).
        """
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta
        self.overlap = int(self.patch_height/7)

    def __call__(self, img):
        # 1) Randomly skip
        if torch.rand(1) > self.mix_prob:
            return img

        # 2) Convert to NumPy float32
        img_np = np.array(img, dtype=np.float32)
        h, w, c = img_np.shape

        # 3) Compute # of patches in each dimension
        num_patches_h = h // self.patch_height
        num_patches_w = w // self.patch_width
        
        # local region of shuffling
        self.M = max(num_patches_h, 4)

        # 4) Precompute a large feather mask for the largest possible patch 
        #    (patch_height + 2*overlap x patch_width + 2*overlap)
        #    We'll slice it down for boundary patches if needed
        feather_mask_full = create_feather_mask(
            self.patch_height + 2*self.overlap,
            self.patch_width  + 2*self.overlap,
            feather_size=4
        )

        # 5) Extract patches
        patches = []
        coords = []
        for i_patch in range(num_patches_h):
            for j_patch in range(num_patches_w):
                start_h, end_h, start_w, end_w = edgelogic(
                    i_patch, j_patch,
                    self.patch_height, self.patch_width,
                    num_patches_h, num_patches_w,
                    self.overlap
                )
                # Clip to image boundaries if needed
                start_h = max(0, start_h)
                start_w = max(0, start_w)
                end_h   = min(h, end_h)
                end_w   = min(w, end_w)

                patch = img_np[start_h:end_h, start_w:end_w]
                patches.append(patch)
                coords.append((start_h, end_h, start_w, end_w))

        # 6) Shuffle & mix patches
        N = len(patches)
        
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
                lambda_ = np.random.beta(self.alpha, self.beta)
                # mixed_patch = lambda_ * reordered_patches[j] + (1 - lambda_) * reordered_patches[shuffled_index]
                mixed_patch = lambda_ * reordered_patches[j] + (1 - lambda_) * reordered_patches[shuffled_index]
                mixed_patches.append(mixed_patch)

        # 7) Prepare output & weight arrays for soft blending
        output = np.zeros_like(img_np, dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        index = 0

        # 8) Blend each patch with the feather mask
        index = 0
        for (row, col) in peano_indices:
            sh, eh, sw, ew = edgelogic(
                row, col,
                self.patch_height, self.patch_width,
                num_patches_h, num_patches_w,
                self.overlap
            )
            patch_mixed = mixed_patches[index]
            ph, pw, _ = patch_mixed.shape

            # Extract the corresponding portion of the big feather mask
            # if patch is smaller near boundaries, slice the mask
            mask_2d = feather_mask_full[:ph, :pw]
            
            # Convert mask to 3 channels if needed
            if c == 1:
                mask_3d = mask_2d[..., None]
            else:
                mask_3d = np.repeat(mask_2d[..., None], c, axis=2)

            # Feathered patch
            patch_feathered = patch_mixed * mask_3d

            # Accumulate in output
            output[sh:eh, sw:ew] += patch_feathered
            weight[sh:eh, sw:ew] += mask_2d
            index += 1

        # 9) Final divide by weight -> smooth blend
        weight = np.clip(weight, 1e-8, None)
        output /= weight[..., None]

        final_output = img_np
        final_output[1:-1,1:-1] = output[1:-1,1:-1]
        # 10) Convert back to uint8

        final_output = np.clip(final_output, 0, 255).astype(np.uint8)
        return Image.fromarray(final_output)
    
class ShufflePatchMixOverlap_l_all():
    def __init__(self, 
                 patch_height, 
                 patch_width, 
                 mix_prob=0.8, 
                 alpha=4.0, 
                 beta=2.0, 
                 overlap=4):
        """
        patch_height, patch_width : Base size of each patch (without overlap).
        overlap                  : Overlap in pixels on each edge.
        mix_prob                 : Probability of applying ShufflePatchMix.
        alpha, beta              : Beta distribution parameters (lambda sampling).
        """
        patch_height_options = [14, 28, 56, 112]
        random_number = np.random.randint(1, len(patch_height_options))
        self.patch_height = patch_height_options[random_number]
        self.patch_width = patch_height_options[random_number] 
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta
        self.overlap = int(self.patch_height/7)

    def __call__(self, img):
        # 1) Randomly skip
        if torch.rand(1) > self.mix_prob:
            return img

        # 2) Convert to NumPy float32
        img_np = np.array(img, dtype=np.float32)
        h, w, c = img_np.shape

        # 3) Compute # of patches in each dimension
        num_patches_h = h // self.patch_height
        num_patches_w = w // self.patch_width
        
        # local region of shuffling
        self.M = max(num_patches_h, 4)

        # 4) Precompute a large feather mask for the largest possible patch 
        #    (patch_height + 2*overlap x patch_width + 2*overlap)
        #    We'll slice it down for boundary patches if needed
        feather_mask_full = create_feather_mask(
            self.patch_height + 2*self.overlap,
            self.patch_width  + 2*self.overlap,
            feather_size=4
        )

        # 5) Extract patches
        patches = []
        coords = []
        for i_patch in range(num_patches_h):
            for j_patch in range(num_patches_w):
                start_h, end_h, start_w, end_w = edgelogic(
                    i_patch, j_patch,
                    self.patch_height, self.patch_width,
                    num_patches_h, num_patches_w,
                    self.overlap
                )
                # Clip to image boundaries if needed
                start_h = max(0, start_h)
                start_w = max(0, start_w)
                end_h   = min(h, end_h)
                end_w   = min(w, end_w)

                patch = img_np[start_h:end_h, start_w:end_w]
                patches.append(patch)
                coords.append((start_h, end_h, start_w, end_w))

        # 6) Shuffle & mix patches
        N = len(patches)
        
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
                lambda_ = np.random.beta(self.alpha, self.beta)
                # mixed_patch = lambda_ * reordered_patches[j] + (1 - lambda_) * reordered_patches[shuffled_index]
                mixed_patch = lambda_ * reordered_patches[j] + (1 - lambda_) * reordered_patches[shuffled_index]
                mixed_patches.append(mixed_patch)

        # 7) Prepare output & weight arrays for soft blending
        output = np.zeros_like(img_np, dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        index = 0

        # 8) Blend each patch with the feather mask
        index = 0
        for (row, col) in peano_indices:
            sh, eh, sw, ew = edgelogic(
                row, col,
                self.patch_height, self.patch_width,
                num_patches_h, num_patches_w,
                self.overlap
            )
            patch_mixed = mixed_patches[index]
            ph, pw, _ = patch_mixed.shape

            # Extract the corresponding portion of the big feather mask
            # if patch is smaller near boundaries, slice the mask
            mask_2d = feather_mask_full[:ph, :pw]
            
            # Convert mask to 3 channels if needed
            if c == 1:
                mask_3d = mask_2d[..., None]
            else:
                mask_3d = np.repeat(mask_2d[..., None], c, axis=2)

            # Feathered patch
            patch_feathered = patch_mixed * mask_3d

            # Accumulate in output
            output[sh:eh, sw:ew] += patch_feathered
            weight[sh:eh, sw:ew] += mask_2d
            index += 1

        # 9) Final divide by weight -> smooth blend
        weight = np.clip(weight, 1e-8, None)
        output /= weight[..., None]

        final_output = img_np
        final_output[1:-1,1:-1] = output[1:-1,1:-1]
        # 10) Convert back to uint8

        final_output = np.clip(final_output, 0, 255).astype(np.uint8)
        return Image.fromarray(final_output)
    
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
        patch_height_options = [14, 28, 56, 112]
        random_number = np.random.randint(1, len(patch_height_options))
        self.patch_height = patch_height_options[random_number]
        self.patch_width = patch_height_options[random_number] 
        self.mix_prob = mix_prob

    def __call__(self, img):
        if  torch.rand(1) > self.mix_prob:
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

