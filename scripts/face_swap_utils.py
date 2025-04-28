import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F

def mix_source_and_target(target,source, alpha=0.5):
    """
    Mixes the source and target images based on the given alpha value.
    :param source: Source image
    :param target: Target image
    :param alpha: Mixing ratio (0.0 to 1.0)
    :return: Mixed image
    """
    
    
    return (1 - alpha) * source + alpha * target

def fft_fusion(noise_A, noise_B, center=16):
    # Apply FFT over H and W for each batch and channel
    fft_A = torch.fft.fft2(noise_A, dim=(-2, -1))
    fft_B = torch.fft.fft2(noise_B, dim=(-2, -1))
    
    fft_A_shift = torch.fft.fftshift(fft_A, dim=(-2, -1))
    fft_B_shift = torch.fft.fftshift(fft_B, dim=(-2, -1))
    
    B, C, H, W = noise_A.shape
    mask = torch.zeros((H, W), device=noise_A.device)
    cx, cy = H // 2, W // 2
    mask[cx - center:cx + center, cy - center:cy + center] = 1
    mask = mask[None, None, :, :]  # shape: (1, 1, H, W)
    
    combined_fft = fft_A_shift * mask + fft_B_shift * (1 - mask)
    combined_fft = torch.fft.ifftshift(combined_fft, dim=(-2, -1))
    
    combined = torch.fft.ifft2(combined_fft, dim=(-2, -1)).real
    return combined



def lpf_fusion(noise_A, noise_B, kernel_size=5, sigma=1.0):
    # Apply LPF over H and W for each batch and channel
    # Structure from low-pass of noise_A
    B, C, H, W = noise_A.shape

    def gaussian_blur(x, kernel_size=5, sigma=1.0):
        # Create Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        grid = coords[None, :]**2 + coords[:, None]**2
        kernel = torch.exp(-grid / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        # Expand to match depthwise conv shape
        kernel = kernel.view(1, 1, kernel_size, kernel_size).to(x.device)
        kernel = kernel.repeat(C, 1, 1, 1)  # Shape: (C, 1, k, k)

        return F.conv2d(x, kernel, padding=kernel_size // 2, groups=C)

    # Low-pass from structure source (A)
    structure_part = gaussian_blur(noise_A, kernel_size, sigma)

    # High-pass from identity source (B)
    blurred_B = gaussian_blur(noise_B, kernel_size, sigma)
    identity_part = noise_B - blurred_B

    # Final fusion
    combined_latent = structure_part + identity_part
    return combined_latent

# def AdaIn_fusion(noise_A, noise_B, alpha=0.71,normalized=True):
#     # Apply AdaIn over H and W for each batch and channel
#     # B, C, H, W = noise_A.shape

#     # Compute mean and std for both noise_A and noise_B
#     mean_A = noise_A.mean(dim=(2,3), keepdim=True)
#     std_A = noise_A.std(dim=(2,3), keepdim=True)
#     mean_B = noise_B.mean(dim=(2,3), keepdim=True)
#     std_B = noise_B.std(dim=(2,3), keepdim=True)

#     # Normalize noise_A
#     normalized_A = (noise_A - mean_A) / (std_A + 1e-5)

#     # Scale and shift with noise_B's statistics
#     fused_noise = normalized_A * std_B + mean_B
    
    
#     # normalize the fused noise
#     if normalized:
#         # Normalize the fused noise
#         fused_noise = (fused_noise) / (fused_noise.std() + 1e-5)
#         return fused_noise
#     else:
#         # Standardize the fused noise
#         return alpha*fused_noise
    
    
def AdaIn_fusion(noise_A, noise_B, alpha=0.71,beta=1.0, normalized=True):
    """
    Apply Adaptive Instance Normalization (AdaIN) fusion.
    noise_A: structure source (B, C, H, W)
    noise_B: identity source (B, C, H, W)
    alpha: blend factor (0 -> only noise_A, 1 -> fully fused)
    normalized: if True, normalize noise_A first
    """
    # Compute per-channel mean and std
    mean_A = noise_A.mean(dim=(2,3), keepdim=True)
    std_A = noise_A.std(dim=(2,3), keepdim=True)
    mean_B = noise_B.mean(dim=(2,3), keepdim=True)
    std_B = noise_B.std(dim=(2,3), keepdim=True)

    if normalized:
        # Normalize noise_A
        normalized_A = (noise_A - mean_A) / (std_A + 1e-5)
    else:
        normalized_A = noise_A  # skip normalization if requested

    # Perform AdaIN: shift and scale to noise_B stats
    fused = normalized_A * (std_B + 1e-5) + mean_B

    # Interpolate between original and fused
    output = (1 - alpha) * noise_A + alpha * fused

    return output*beta

def AdaIn_fusion_for_attn(noise_A, noise_B, alpha=0.71,normalized=True):
    # Apply AdaIn over H and W for each batch and channel
    # B, C, H, W = noise_A.shape

    # Compute mean and std for both noise_A and noise_B
    mean_A = noise_A.mean(dim= -1, keepdim=True)
    std_A = noise_A.std(dim=-1, keepdim=True)
    mean_B = noise_B.mean(dim=-1, keepdim=True)
    std_B = noise_B.std(dim=-1, keepdim=True)

    # Normalize noise_A
    normalized_A = (noise_A - mean_A) / (std_A + 1e-5)

    # Scale and shift with noise_B's statistics
    fused_noise = normalized_A * std_B + mean_B
    
    
    # normalize the fused noise
    if normalized:
        # Normalize the fused noise
        fused_noise = (fused_noise) / (fused_noise.std() + 1e-5)
        return fused_noise
    else:
        # Standardize the fused noise
        return alpha*fused_noise

def plot_fft_3d(fft, title=None):
    import matplotlib.pyplot as plt
    import numpy as np

    # Convert to numpy for plotting
    fft_np = fft.cpu().numpy()
    
    # Plot the magnitude spectrum
    plt.figure(figsize=(10, 10))
    plt.imshow(np.abs(fft_np[0, 0]), cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()

 
def plot_fft_3d(latent_tensor, batch_idx=0, channel_idx=0, log_scale=True,save_path="out.png"):
    """
    latent_tensor: torch.Tensor of shape (B, C, H, W)
    batch_idx: index of batch to visualize
    channel_idx: index of channel to visualize
    log_scale: whether to apply log(1 + magnitude) for better visualization
    """
    # Select one channel from the batch
    selected = latent_tensor[batch_idx, channel_idx]  # shape: (H, W)
    
    # Compute 2D FFT and shift
    fft2d = torch.fft.fft2(selected)
    fft2d_shifted = torch.fft.fftshift(fft2d)
    magnitude = torch.abs(fft2d_shifted)

    # Optional log scale for better dynamic range
    if log_scale:
        magnitude = torch.log1p(magnitude)

    # Prepare mesh grid
    H, W = selected.shape
    X, Y = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

    # Plot 3D surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X.numpy(), Y.numpy(), magnitude.cpu().numpy(), cmap='viridis')

    ax.set_title(f"3D FFT Spectrum (B={batch_idx}, C={channel_idx})")
    ax.set_xlabel("Height")
    ax.set_ylabel("Width")
    ax.set_zlabel("Magnitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()
 
 
 
 
 
 
 
 
 
 
 # standardize start code
# start_code = (start_code - start_code.mean()) / start_code.std()

# check this hyper parameter
# start_code=(x_noisy_target+x_noisy_src)/1.41

# start_code=x_noisy_src


# Optional
# inp_mask=test_model_kwargs['inpaint_mask']
# inp_mask=inp_mask.repeat(1, 4, 1, 1)
# start_code[inp_mask==1.0]=x_noisy_target[inp_mask==1.0]

# start_code=x_noisy_target
# start_code_noise=torch.randn_like(start_code)
# alpha = 0.5  # Adjust this
# start_code = alpha * start_code + (1 - alpha) * torch.randn_like(start_code)
# start_code=start_code/0.7
# noise = torch.randn_like(z)

# x_noisy = model.q_sample(x_start=z, t=t, noise=noise)
# start_code = x_noisy