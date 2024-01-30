from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T

def un_norm_clip(x1):
    x = x1*1.0

    x[0,:,:] = x[0,:,:] * 0.26862954 + 0.48145466
    x[1,:,:] = x[1,:,:] * 0.26130258 + 0.4578275
    x[2,:,:] = x[2,:,:] * 0.27577711 + 0.40821073
    
    return x

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
    # x = x.clone()
    return (x+1.0)/2.0
 
def save_clip_img(img2, path="/home/sanoojan/Paint_for_swap/Debug/OUT.png",clip=True):
    if clip:
        img=un_norm_clip(img2)
    else:
        img=torch.clamp(un_norm(img2), min=0.0, max=1.0)
    img = img.cpu().numpy().transpose((1, 2, 0))
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)
    
    
    
def read_image_to_tensor():
    # read image from disk using PIL
    img = Image.open("/home/sanoojan/Paint_for_swap/Debug/0rec.png")
    # convert image to torch tensor
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    # img_tensor = img_tensor
    # normalize image
    img_tensor = T.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])(img_tensor)
    return img_tensor




x=read_image_to_tensor()
save_clip_img(x, path="/home/sanoojan/Paint_for_swap/Debug/OUT1.png",clip=True)
save_clip_img(x, path="/home/sanoojan/Paint_for_swap/Debug/OUT2.png",clip=False)
save_clip_img(x, path="/home/sanoojan/Paint_for_swap/Debug/OUT3.png",clip=True)
save_clip_img(x, path="/home/sanoojan/Paint_for_swap/Debug/OUT4.png",clip=False)
print("Reading image")
print("Image read")