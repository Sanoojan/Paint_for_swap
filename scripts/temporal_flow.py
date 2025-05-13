import torch
# from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
device = "cuda" if torch.cuda.is_available() else "cpu"
import torchvision
import numpy as np
from PIL import Image

# from raft import RAFT
# from utils.utils import InputPadder
import argparse
import torch.nn.functional as F
import kornia

# Load RAFT model
# def load_raft_model():
#     args = argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False)
#     model = torch.nn.DataParallel(RAFT(args))
#     model.load_state_dict(torch.load('raft-things.pth'))  # Download from RAFT repo
#     model = model.module.eval().cuda()
#     return model

# raft_model = load_raft_model()


model = raft_large(pretrained=True, progress=False).to(device)
model = model.eval()


# Compute optical flow
@torch.no_grad()
def compute_flow(img1, img2, model):
    # padder = InputPadder(img1.shape)
    # img1, img2 = padder.pad(img1, img2)
    flow_up = model(img1, img2, num_flow_updates=20)
    return flow_up[-1]  # [B, 2, H, W]

# Warp image using flow
def warp_image(img, flow):
    # img: [B, C, H, W], flow: [B, 2, H, W]
    B, C, H, W = img.size()
    grid = kornia.utils.create_meshgrid(H, W, normalized_coordinates=False).to(img.device)  # [1, H, W, 2]
    grid = grid.permute(0, 3, 1, 2)  # [1, 2, H, W]
    vgrid = grid + flow

    # Normalize to [-1, 1] for grid_sample
    vgrid[:, 0] = 2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0
    vgrid[:, 1] = 2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)  # [B, H, W, 2]

    output = F.grid_sample(img, vgrid, align_corners=True, padding_mode='border')
    return output

def decode_latents(latents, decode_fn):
    # latents: (B, 4, H, W) → images: (B, 3, H, W)
        img = decode_fn(latents)
        return img
 
def encode_latents(images, encode_fn,first_stage_fn):
    # images: (B, 3, H, W) → latents: (B, 4, H, W)
    enc= encode_fn(images)  
    return first_stage_fn(enc).detach()  

def un_norm_clip(x1):
    x = x1*1.0 # to avoid changing the original tensor or clone() can be used
    reduce=False
    if len(x.shape)==3:
        x = x.unsqueeze(0)
        reduce=True
    x[:,0,:,:] = x[:,0,:,:] * 0.26862954 + 0.48145466
    x[:,1,:,:] = x[:,1,:,:] * 0.26130258 + 0.4578275
    x[:,2,:,:] = x[:,2,:,:] * 0.27577711 + 0.40821073
    
    if reduce:
        x = x.squeeze(0)
    return x

def un_norm(x):
    return (x+1.0)/2.0

def save_clip_img(img, path,clip=True):
    if clip:
        img=un_norm_clip(img)
    else:
        img=torch.clamp(un_norm(img), min=0.0, max=1.0)
    img = img.cpu().numpy().transpose((1, 2, 0))
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)

def save_flow_img(flow, filename):
    # Convert flow to RGB image
    flow_rgb = flow_to_image(flow.cpu())  # [B, 3, H, W]
    img= flow_rgb[0].permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(filename)
    
   
@torch.no_grad()
def batch_flow_align(x_prev, x_prev_recon, decode_fn,encode_fn,first_stage_fn, alpha=0.5):
    """
    x_prev:        (B, 4, H, W) - original sample
    x_prev_recon:  (B, 4, H, W) - flow reference
    """
    B = x_prev.shape[0]
    aligned_x_prev = x_prev.clone()
    
    rgb_recons = decode_latents(x_prev_recon, decode_fn)  # (B, 3, H, W)
    rgb_prev    = decode_latents(x_prev, decode_fn)
    
    # return x_prev
    
    # return encode_latents(rgb_prev, encode_fn,first_stage_fn)

    for i in range(B - 1):
        # Get RGB frames
        frame1 = rgb_recons[i].unsqueeze(0)  # (1, 3, H, W)
        frame2 = rgb_recons[i + 1].unsqueeze(0)

        # Flow from frame1 → frame2
        flow = compute_flow(frame2, frame1, model)  # (1, 2, H, W)

        # Warp x_prev[i] to match x_prev[i+1]
        warped_rgb = warp_image(rgb_prev[i].unsqueeze(0), flow)
        aligned = alpha * rgb_prev[i + 1] + (1 - alpha) * warped_rgb.squeeze(0)
        rgb_prev[i + 1] = aligned
        aligned_latent = encode_latents(aligned.unsqueeze(0), encode_fn, first_stage_fn)
        aligned_x_prev[i + 1] = aligned_latent.squeeze(0)
        
        # warped = warp_image(x_prev[i].unsqueeze(0), flow)  # (1, 4, H, W)

        # # Blend with current sample
        # aligned = alpha * x_prev[i + 1] + (1 - alpha) * warped.squeeze(0)
        # aligned_x_prev[i + 1] = aligned  # Replace or store elsewhere if needed
    
    

    return aligned_x_prev


@torch.no_grad()
def batch_flow_align_latent(x_prev, x_prev_recon, decode_fn,encode_fn,first_stage_fn, alpha=0.5):
    """
    x_prev:        (B, 4, H, W) - original sample
    x_prev_recon:  (B, 4, H, W) - flow reference
    """
    B = x_prev.shape[0]
    aligned_x_prev = x_prev.clone()
    
    rgb_recons = decode_latents(x_prev_recon, decode_fn)  # (B, 3, H, W)
    # rgb_prev    = decode_latents(x_prev, decode_fn)
    
    # return x_prev
    
    # return encode_latents(rgb_prev, encode_fn,first_stage_fn)

    for i in range(B - 1):
        # Get RGB frames
        frame1 = rgb_recons[i].unsqueeze(0)  # (1, 3, H, W)
        frame2 = rgb_recons[i + 1].unsqueeze(0)
        
        # resize to 64,64
        frame1 = F.interpolate(frame1, size=(64, 64), mode='bilinear', align_corners=False)
        frame2 = F.interpolate(frame2, size=(64, 64), mode='bilinear', align_corners=False)

        # Flow from frame1 → frame2
        flow = compute_flow(frame2, frame1, model)  # (1, 2, H, W)

        # Warp x_prev[i] to match x_prev[i+1]
        warped_latents = warp_image(x_prev[i].unsqueeze(0), flow)
        aligned = alpha * x_prev[i + 1] + (1 - alpha) * warped_latents.squeeze(0)
        x_prev[i + 1] = aligned
        # aligned_latent = encode_latents(aligned.unsqueeze(0), encode_fn, first_stage_fn)
        # aligned_x_prev[i + 1] = aligned_latent.squeeze(0)
        
        # warped = warp_image(x_prev[i].unsqueeze(0), flow)  # (1, 4, H, W)

        # # Blend with current sample
        # aligned = alpha * x_prev[i + 1] + (1 - alpha) * warped.squeeze(0)
        # aligned_x_prev[i + 1] = aligned  # Replace or store elsewhere if needed
    
    

    return x_prev

def test_enc(x_prev, x_prev_recon, decode_fn,encode_fn,first_stage_fn, alpha=0.5):
    x_prev=encode_latents(x_prev, encode_fn,first_stage_fn)
    x_prev=decode_latents(x_prev, decode_fn)
    
    return x_prev