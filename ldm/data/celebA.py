from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image,ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import bezier

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision.utils import save_image

def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


#####


# 19 attributes in total, skin-1,nose-2,...cloth-18, background-0
celelbAHQ_label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                        'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r',
                        'neck_l', 'neck', 'cloth']

# face-parsing.PyTorch also includes 19 attributes，but with different permutation
face_parsing_PyTorch_label_list = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
                                    'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 
                                    'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 
                                    'cloth', 'hair', 'hat']  # skin-1 l_brow-2 ...
 
# 9 attributes with left-right aggrigation
faceParser_label_list = ['background', 'mouth', 'eyebrows', 'eyes', 'hair', 
                         'nose', 'skin', 'ears', 'belowface']

# 12 attributes with left-right aggrigation
faceParser_label_list_detailed = ['background', 'lip', 'eyebrows', 'eyes', 'hair', 
                                  'nose', 'skin', 'ears', 'belowface', 'mouth', 
                                  'eye_glass', 'ear_rings']

TO_TENSOR = transforms.ToTensor()
MASK_CONVERT_TF = transforms.Lambda(
    lambda celebAHQ_mask: __celebAHQ_masks_to_faceParser_mask(celebAHQ_mask))

MASK_CONVERT_TF_DETAILED = transforms.Lambda(
    lambda celebAHQ_mask: __celebAHQ_masks_to_faceParser_mask_detailed(celebAHQ_mask))


NORMALIZE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def get_transforms(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)



def __celebAHQ_masks_to_faceParser_mask_detailed(celebA_mask):
    """Convert the semantic image of CelebAMaskHQ to reduced categories (12-class). 

    Args:
        mask (PIL image): with shape [H,W]
    Return:
        aggrigated mask, with same shape [H,W] but the number of segmentation classes is less
    """
    # 19 attributes in total, skin-1,nose-2,...cloth-18, background-0
    celelbAHQ_label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                            'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                            'u_lip', 'l_lip', 'hair', 'hat', 'ear_r',
                            'neck_l', 'neck', 'cloth']# 12 attributes with left-right aggrigation
    faceParser_label_list_detailed = ['background', 'lip', 'eyebrows', 'eyes', 'hair', 
                                    'nose', 'skin', 'ears', 'belowface', 'mouth', 
                                  'eye_glass', 'ear_rings']

    converted_mask = np.zeros_like(celebA_mask)

    backgorund = np.equal(celebA_mask, 0)
    converted_mask[backgorund] = 0

    lip = np.logical_or(np.equal(celebA_mask, 11), np.equal(celebA_mask, 12))
    converted_mask[lip] = 1

    eyebrows = np.logical_or(np.equal(celebA_mask, 6),
                             np.equal(celebA_mask, 7))
    converted_mask[eyebrows] = 2

    eyes = np.logical_or(np.equal(celebA_mask, 4), np.equal(celebA_mask, 5))
    converted_mask[eyes] = 3

    hair = np.equal(celebA_mask, 13)
    converted_mask[hair] = 4

    nose = np.equal(celebA_mask, 2)
    converted_mask[nose] = 5

    skin = np.equal(celebA_mask, 1)
    # print('skin', np.sum(skin))
    converted_mask[skin] = 6

    ears = np.logical_or(np.equal(celebA_mask, 8), np.equal(celebA_mask, 9))
    converted_mask[ears] = 7

    belowface = np.equal(celebA_mask, 17)
    converted_mask[belowface] = 8
    
    mouth = np.equal(celebA_mask, 10)   
    converted_mask[mouth] = 9

    eye_glass = np.equal(celebA_mask, 3)
    converted_mask[eye_glass] = 10
    
    ear_rings = np.equal(celebA_mask, 15)
    converted_mask[ear_rings] = 11
    
    return converted_mask

def __celebAHQ_masks_to_faceParser_mask(celebA_mask):
    """Convert the semantic image of CelebAMaskHQ to reduced categories (9-class). 

    Args:
        mask (PIL image): with shape [H,W]
    Return:
        aggrigated mask, with same shape [H,W] but the number of segmentation classes is less
    """

    assert len(celebA_mask.size) == 2, "The provided mask should be with [H,W] format"

    converted_mask = np.zeros_like(celebA_mask)

    backgorund = np.equal(celebA_mask, 0)
    converted_mask[backgorund] = 0

    mouth = np.logical_or(
        np.logical_or(np.equal(celebA_mask, 10), np.equal(celebA_mask, 11)),
        np.equal(celebA_mask, 12)
    )
    converted_mask[mouth] = 1

    eyebrows = np.logical_or(np.equal(celebA_mask, 6),
                             np.equal(celebA_mask, 7))
    converted_mask[eyebrows] = 2

    eyes = np.logical_or(np.equal(celebA_mask, 4), np.equal(celebA_mask, 5))
    converted_mask[eyes] = 3

    hair = np.equal(celebA_mask, 13)
    converted_mask[hair] = 4

    nose = np.equal(celebA_mask, 2)
    converted_mask[nose] = 5

    skin = np.equal(celebA_mask, 1)
    converted_mask[skin] = 6

    ears = np.logical_or(np.equal(celebA_mask, 8), np.equal(celebA_mask, 9))
    converted_mask[ears] = 7

    belowface = np.equal(celebA_mask, 17)
    converted_mask[belowface] = 8

    return converted_mask




class CelebAdataset(data.Dataset):
    def __init__(self,state,arbitrary_mask_percent=0,load_vis_img=False,label_transform=None,fraction=1.0,**args
        ):
        self.label_transform=label_transform
        self.fraction=fraction
        self.load_vis_img=load_vis_img
        self.state=state
        self.args=args
        self.arbitrary_mask_percent=arbitrary_mask_percent
        self.kernel = np.ones((1, 1), np.uint8)
        self.random_trans=A.Compose([
            A.Resize(height=224,width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3)
            ])
        # bad_list=[
        #     '1af17f3d912e9aac.txt',
        #     '1d5ef05c8da80e31.txt',
        #     '3095084b358d3f2d.txt',
        #     '3ad7415a11ac1f5e.txt',
        #     '42a30d8f8fba8b40.txt',
        #     '1366cde3b480a15c.txt',
        #     '03a53ed6ab408b9f.txt'
        # ]
        
        self.bbox_path_list=[]
        if state == "train":
            self.imgs = sorted([osp.join(args['dataset_dir'], "CelebA-HQ-img", "%d.jpg"%idx) for idx in range(28000)])
            # self.labels = ([osp.join(self.root, "CelebA-HQ-mask", "%d"%int(idx/2000) ,'{0:0=5d}'.format(idx)+'_skin.png') for idx in range(28000)])
            self.labels =  sorted([osp.join(args['dataset_dir'], "CelebA-HQ-mask/Overall_mask", "%d.png"%idx) for idx in range(28000)]) 
            self.labels_vis =  sorted([osp.join(args['dataset_dir'], "vis", "%d.png"%idx) for idx in range(28000)]) if self.load_vis_img else None
        elif state == "validation":
            self.imgs = sorted([osp.join(args['dataset_dir'], "CelebA-HQ-img", "%d.jpg"%idx) for idx in range(28000, 30000)])
            # self.labels = ([osp.join(self.root, "CelebA-HQ-mask", "%d"%int(idx/2000) ,'{0:0=5d}'.format(idx)+'_skin.png') for idx in range(28000, 30000)])
            self.labels =  sorted([osp.join(args['dataset_dir'], "CelebA-HQ-mask/Overall_mask", "%d.png"%idx) for idx in range(28000, 30000)]) 
            self.labels_vis =  sorted([osp.join(args['dataset_dir'], "vis", "%d.png"%idx) for idx in range(28000, 30000)]) if self.load_vis_img else None
        else:
            self.imgs = sorted([osp.join(args['dataset_dir'], "CelebA-HQ-img", "%d.jpg"%idx) for idx in range(28000, 30000)])
            # self.labels = ([osp.join(self.root, "CelebA-HQ-mask", "%d"%int(idx/2000) ,'{0:0=5d}'.format(idx)+'_skin.png') for idx in range(28000, 30000)])
            self.labels =  sorted([osp.join(args['dataset_dir'], "CelebA-HQ-mask/Overall_mask", "%d.png"%idx) for idx in range(28000, 30000)]) 
            self.labels_vis =  sorted([osp.join(args['dataset_dir'], "vis", "%d.png"%idx) for idx in range(28000, 30000)]) if self.load_vis_img else None
        
        self.imgs= self.imgs[:int(len(self.imgs)*self.fraction)]
        self.labels= self.labels[:int(len(self.labels)*self.fraction)]
        self.labels_vis= self.labels_vis[:int(len(self.labels_vis)*self.fraction)]  if self.load_vis_img else None

        if self.load_vis_img:
            assert len(self.imgs) == len(self.labels) == len(self.labels_vis)
        else:
            assert len(self.imgs) == len(self.labels)

        # image pairs indices
        self.indices = np.arange(len(self.imgs))
        self.length=len(self.indices)

    def load_single_image(self, index):
        """Load one sample for training, inlcuding 
            - the image, 
            - the semantic image, 
            - the corresponding visualization image

        Args:
            index (int): index of the sample
        Return:
            img: RGB image
            label: seg mask
            label_vis: visualization of the seg mask
        """
        img = self.imgs[index]
        img = Image.open(img).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)

        label = self.labels[index]
        # print(label)
        label = Image.open(label).convert('L')
        # breakpoint()
        # label2=TO_TENSOR(label)
        # save_image(label2, str(index)+'_label.png')
        # save_image(img, str(index)+'_img.png')  
        
        if self.label_transform is not None:
            label= self.label_transform(label)
 

        if self.load_vis_img:
            label_vis = self.labels_vis[index]
            label_vis = Image.open(label_vis).convert('RGB')
            label_vis = TO_TENSOR(label_vis)
        else:
            label_vis = -1  # unified interface
        # save_image(label, str(index)+'_label.png')
        # save_image(img, str(index)+'_img.png')  

        return img, label, label_vis
  

    
    def __getitem__(self, index):
        # bbox_path=self.bbox_path_list[index]
        # file_name=os.path.splitext(os.path.basename(bbox_path))[0]+'.jpg'
        # dir_name=bbox_path.split('/')[-2]
        # img_path=os.path.join('dataset/open-images/images',dir_name,file_name)


        # bbox_list=[]
        # with open(bbox_path) as f:
        #     line=f.readline()
        #     while line:
        #         line_split=line.strip('\n').split(" ")
        #         bbox_temp=[]
        #         for i in range(4):
        #             bbox_temp.append(int(float(line_split[i])))
        #         bbox_list.append(bbox_temp)
        #         line=f.readline()
        # bbox=random.choice(bbox_list)
        # img_p = Image.open(img_path).convert("RGB")
        
        img_path = self.imgs[index]
        img_p = Image.open(img_path).convert('RGB')
        # if self.img_transform is not None:
        #     img = self.img_transform(img)

        label = self.labels[index]
        label = Image.open(label).convert('L')
        # Assuming that 'label' is your binary mask (black and white image)
        label = np.array(label)  # Convert the label to a NumPy array if it's not already

        # Find the coordinates of the non-zero (white) pixels in the mask
        non_zero_coords = np.column_stack(np.where(label == 1))

        # Find the minimum and maximum x and y coordinates to get the bounding box
        min_x, min_y = np.min(non_zero_coords, axis=0)
        max_x, max_y = np.max(non_zero_coords, axis=0)

        # Add padding if needed
        padding = 0
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(img_p.size[0], max_x + padding)
        max_y = min(img_p.size[1], max_y + padding)

        # The bounding box coordinates are now (min_x, min_y, max_x, max_y)
        # Scale the bounding box coordinates to match the image size (1024x1024)
        min_x *= 2
        min_y *= 2
        max_x *= 2
        max_y *= 2
        bbox = [min_x, min_y, max_x, max_y]
        
        if self.label_transform is not None:
            label= self.label_transform(label)
 

        if self.load_vis_img:
            label_vis = self.labels_vis[index]
            label_vis = Image.open(label_vis).convert('RGB')
            label_vis = TO_TENSOR(label_vis)
        else:
            label_vis = -1  # unified interface
        
        # img_p, label, label_vis = self.load_single_image(index)
        # bbox=[30,50,60,100]
   
        ### Get reference image
        bbox_pad=copy.copy(bbox)
        bbox_pad[0]=bbox[0]-min(10,bbox[0]-0)
        bbox_pad[1]=bbox[1]-min(10,bbox[1]-0)
        bbox_pad[2]=bbox[2]+min(10,img_p.size[0]-bbox[2])
        bbox_pad[3]=bbox[3]+min(10,img_p.size[1]-bbox[3])
        img_p_np=cv2.imread(img_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        ref_image_tensor=img_p_np[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2],:]
        ref_image_tensor=self.random_trans(image=ref_image_tensor)
        ref_image_tensor=Image.fromarray(ref_image_tensor["image"])
        ref_image_tensor=get_tensor_clip()(ref_image_tensor)



        ### Generate mask
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size

        extended_bbox=copy.copy(bbox)
        left_freespace=bbox[0]-0
        right_freespace=W-bbox[2]
        up_freespace=bbox[1]-0
        down_freespace=H-bbox[3]
        extended_bbox[0]=bbox[0]-random.randint(0,int(0.4*left_freespace))
        extended_bbox[1]=bbox[1]-random.randint(0,int(0.4*up_freespace))
        extended_bbox[2]=bbox[2]+random.randint(0,int(0.4*right_freespace))
        extended_bbox[3]=bbox[3]+random.randint(0,int(0.4*down_freespace))

        prob=random.uniform(0, 1)
        if prob<self.arbitrary_mask_percent:
            mask_img = Image.new('RGB', (W, H), (255, 255, 255)) 
            bbox_mask=copy.copy(bbox)
            extended_bbox_mask=copy.copy(extended_bbox)
            top_nodes = np.asfortranarray([
                            [bbox_mask[0],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[2]],
                            [bbox_mask[1], extended_bbox_mask[1], bbox_mask[1]],
                        ])
            down_nodes = np.asfortranarray([
                    [bbox_mask[2],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[0]],
                    [bbox_mask[3], extended_bbox_mask[3], bbox_mask[3]],
                ])
            left_nodes = np.asfortranarray([
                    [bbox_mask[0],extended_bbox_mask[0] , bbox_mask[0]],
                    [bbox_mask[3], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[1]],
                ])
            right_nodes = np.asfortranarray([
                    [bbox_mask[2],extended_bbox_mask[2] , bbox_mask[2]],
                    [bbox_mask[1], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[3]],
                ])
            top_curve = bezier.Curve(top_nodes,degree=2)
            right_curve = bezier.Curve(right_nodes,degree=2)
            down_curve = bezier.Curve(down_nodes,degree=2)
            left_curve = bezier.Curve(left_nodes,degree=2)
            curve_list=[top_curve,right_curve,down_curve,left_curve]
            pt_list=[]
            random_width=5
            for curve in curve_list:
                x_list=[]
                y_list=[]
                for i in range(1,19):
                    if (curve.evaluate(i*0.05)[0][0]) not in x_list and (curve.evaluate(i*0.05)[1][0] not in y_list):
                        pt_list.append((curve.evaluate(i*0.05)[0][0]+random.randint(-random_width,random_width),curve.evaluate(i*0.05)[1][0]+random.randint(-random_width,random_width)))
                        x_list.append(curve.evaluate(i*0.05)[0][0])
                        y_list.append(curve.evaluate(i*0.05)[1][0])
            mask_img_draw=ImageDraw.Draw(mask_img)
            mask_img_draw.polygon(pt_list,fill=(0,0,0))
            mask_tensor=get_tensor(normalize=False, toTensor=True)(mask_img)[0].unsqueeze(0)
        else:
            mask_img=np.zeros((H,W))
            mask_img[extended_bbox[1]:extended_bbox[3],extended_bbox[0]:extended_bbox[2]]=1
            mask_img=Image.fromarray(mask_img)
            mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)

        ### Crop square image
        if W > H:
            left_most=extended_bbox[2]-H
            if left_most <0:
                left_most=0
            right_most=extended_bbox[0]+H
            if right_most > W:
                right_most=W
            right_most=right_most-H
            if right_most<= left_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
            else:
                left_pos=random.randint(left_most,right_most) 
                free_space=min(extended_bbox[1]-0,extended_bbox[0]-left_pos,left_pos+H-extended_bbox[2],H-extended_bbox[3])
                random_free_space=random.randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
                mask_tensor_cropped=mask_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
        
        elif  W < H:
            upper_most=extended_bbox[3]-W
            if upper_most <0:
                upper_most=0
            lower_most=extended_bbox[1]+W
            if lower_most > H:
                lower_most=H
            lower_most=lower_most-W
            if lower_most<=upper_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
            else:
                upper_pos=random.randint(upper_most,lower_most) 
                free_space=min(extended_bbox[1]-upper_pos,extended_bbox[0]-0,W-extended_bbox[2],upper_pos+W-extended_bbox[3])
                random_free_space=random.randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
                mask_tensor_cropped=mask_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
        else:
            image_tensor_cropped=image_tensor
            mask_tensor_cropped=mask_tensor

        image_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(image_tensor_cropped)
        mask_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(mask_tensor_cropped)
        inpaint_tensor_resize=image_tensor_resize*mask_tensor_resize
        
        # save_image(image_tensor_resize, "Train_data_images/"+str(index)+'_image_tensor_resize.png')
        # save_image(inpaint_tensor_resize, "Train_data_images/"+ str(index)+'_inpaint_tensor_resize.png')
        # save_image(mask_tensor_resize, "Train_data_images/"+ str(index)+'_mask_tensor_resize.png')
        # save_image(ref_image_tensor,  "Train_data_images/"+str(index)+'_ref_image_tensor.png')
        
        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_tensor_resize,"ref_imgs":ref_image_tensor}

    # def __getitem__(self, idx):
    #     index = self.indices[idx]
    #     img, label, label_vis = self.load_single_image(index)
    #     if self.flip_p > 0:
    #         if random.random() < self.flip_p:
    #             img = TF.hflip(img)
    #             label = TF.hflip(label)
           
    #     return img, label, label_vis

    def __len__(self):
        return self.length
    
    



        






