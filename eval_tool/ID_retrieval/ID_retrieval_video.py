"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF
from PIL import Image
import re
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from src.Face_models.encoders.model_irse import Backbone
# import clip
import torchvision


try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x
import cv2
import albumentations as A
import torch.nn as nn
from natsort import natsorted

# from inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=1,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dataset', type=str, default='celeba',help='Dataset to use')
parser.add_argument('--mask', type=bool, default=True,
                    help='whether to use mask or not')
# parser.add_argument('--dims', type=int, default=2048,
#                     choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
#                     help=('Dimensionality of Inception features to use. '
#                           'By default, uses pool3 features'))
parser.add_argument('path', type=str, nargs=4,
                    default=['dataset/FaceData/CelebAMask-HQ/CelebA-HQ-img', 'results/test_bench/results','dataset/FaceData/CelebAMask-HQ/src_mask','dataset/FaceData/CelebAMask-HQ/target_mask'],
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

parser.add_argument('--print_sim', type=bool, default=False,)
parser.add_argument('--arcface', type=bool, default=False)
parser.add_argument('--target_label_path', type=str)
parser.add_argument('--results_subfolder', type=str)
parser.add_argument('--target_subfolder', type=str)
parser.add_argument('--number_of_images', type=int, default=None)
parser.add_argument('--crop_coordinates', type=str,default='./crop_coordinates')

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

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

def invert_pillow_perspective(coeff):
    """
    Takes 8-tuple PIL-style perspective coefficients and returns the inverse coefficients.
    """
    # Append 1.0 to get a full 3x3 homography matrix
    M = np.append(coeff, 1.0).reshape(3, 3)
    
    # Invert the matrix
    M_inv = np.linalg.inv(M)
    
    # Normalize so bottom-right is 1.0
    M_inv /= M_inv[2, 2]
    
    # Convert back to 8-coefficient PIL format
    inv_coeff = np.concatenate([M_inv[0, :], M_inv[1, :], M_inv[2, :2]])
    return inv_coeff.astype(np.float32)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

class IDLoss(nn.Module):
    def __init__(self,multiscale=True):
        super(IDLoss, self).__init__()
        # print('Loading ResNet ArcFace')
        # self.opts = opts 
        self.multiscale = multiscale
        self.face_pool_1 = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
    
        self.facenet.load_state_dict(torch.load("Other_dependencies/arcface/model_ir_se50.pth"))
        
        self.face_pool_2 = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        
        self.set_requires_grad(False)
            
    def set_requires_grad(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag
    
    def extract_feats(self, x,clip_img=False):
        # breakpoint()
        if clip_img:
            x = un_norm_clip(x)
            x = TFF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        x = self.face_pool_1(x)  if x.shape[2]!=256 else  x # (1) resize to 256 if needed
        x = x[:, :, 35:223, 32:220]  # (2) Crop interesting region
        x = self.face_pool_2(x) # (3) resize to 112 to fit pre-trained model
        # breakpoint()
        x_feats = self.facenet(x, multi_scale=self.multiscale )

        return x_feats

    def forward(self, x,clip_img=False):
        x_feats_ms = self.extract_feats(x,clip_img=clip_img)
        return x_feats_ms[-1]
   

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None,coordinate_path=None):
        self.files = files
        self.transforms = transforms
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.coordinate_path = coordinate_path
        if coordinate_path is not None:
            self.coordinates = np.load(coordinate_path)
        else:
            self.coordinates = None
        
        # _, self.preprocess = clip.load("ViT-B/32", device=device)
        # self.preprocess
        # eval_transform = transforms.Compose([transforms.ToTensor(),
        #                                  transforms.Resize(112),
        #                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        if self.coordinates is not None:
            # Load the coordinates for the current image
            coord=self.coordinates[i]
     
            inv_coeff=invert_pillow_perspective(coord)
            # Load the image and apply the perspective transformation
            image = Image.open(path).convert('RGB')
            image = image.transform((1024,1024), Image.PERSPECTIVE, inv_coeff, Image.BILINEAR)
            image=get_tensor()(image.resize((112,112))).unsqueeze(0)
        else:
            image = get_tensor()(Image.open(path).convert('RGB').resize((112,112))).unsqueeze(0)
        return image


class MaskedImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files,maskfiles=None, transforms=None,data_name="celeba",coordinate_path=None):
        self.coordinate_path = coordinate_path
        if coordinate_path is not None:
            self.coordinates = np.load(coordinate_path)
        else:
            self.coordinates = None
        self.files = files
        self.maskfiles = maskfiles  
        self.transforms = transforms
        self.trans=A.Compose([
            A.Resize(height=112,width=112)])
        self.data_name=data_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # _, self.preprocess = clip.load("ViT-B/32", device=device)
        # self.preprocess
        # eval_transform = transforms.Compose([transforms.ToTensor(),
        #                                  transforms.Resize(112),
        #                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, i):
        path = self.files[i]
        # image=Image.open(path).convert('RGB')
        # ref_img_path = self.ref_imgs[index]
        # print(path)
        
        if self.coordinates is not None:
            # breakpoint()
            # coord=self.coordinates[i
            coord=self.coordinates[i]
            image = Image.open(path).convert('RGB')
            inv_coeff=invert_pillow_perspective(coord)
            image=image.transform((1024,1024), method=Image.PERSPECTIVE, data=inv_coeff,resample=Image.BILINEAR)
            
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        else:
        
            image=cv2.imread(str(path))
            # ref_img = Image.open(ref_img_path).convert('RGB').resize((224,224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
       

        if self.data_name=="celeba":
            preserve = [1,2,4,5,8,9 ,6,7,10,11,12 ]
        elif self.data_name=="ffhq":
            preserve = [1,2,3,5,6,7,9]
        elif self.data_name=="ff++":
            preserve = [1,2,4,5,8,9 ]
        else:
            preserve=None # No mask
        # preserve = [1,2,4,5,8,9 ,6,7,10,11,12 ] # CelebA-HQ
        # preserve = [1,2,3,5,6,7,9]  # FFHQ or FF++
        # print("preserve:",preserve)
        # preserve = [1,2,4,5,8,9 ]
        if preserve is not None:
            
            mask_path = self.maskfiles[i]
            ref_mask_img = Image.open(mask_path).convert('L')
            ref_mask_img = np.array(ref_mask_img)  # Convert the label to a NumPy array if it's not already
            ref_mask= np.isin(ref_mask_img, preserve)

            # Create a converted_mask where preserved values are set to 255
            ref_converted_mask = np.zeros_like(ref_mask_img)
            ref_converted_mask[ref_mask] = 255
            ref_converted_mask=Image.fromarray(ref_converted_mask).convert('L')
            # convert to PIL image
            
            reference_mask_tensor=get_tensor(normalize=False, toTensor=True)(ref_converted_mask)
            mask_ref=TF.Resize((112,112))(reference_mask_tensor)
            ref_img=self.trans(image=image)
            ref_img=Image.fromarray(ref_img["image"])
            ref_img=get_tensor()(ref_img)
            ref_img=ref_img*mask_ref
            image = ref_img.unsqueeze(0)
        else:
            ref_img=self.trans(image=image)
            ref_img=Image.fromarray(ref_img["image"])
            ref_img=get_tensor()(ref_img)
            image = ref_img.unsqueeze(0)
        
        
        # ref_mask_img_r = ref_converted_mask.resize(image.shape[1::-1], Image.NEAREST)
        # ref_mask_img_r = np.array(ref_mask_img_r)
        # image[ref_mask_img_r==0]=0
        
        # image=self.trans(image=image)
        # image=Image.fromarray(image["image"])
        # image=get_tensor()(image)
        
        
        # # ref_img=Image.fromarray(ref_img)
        
        # # ref_img=get_tensor_clip()(ref_img)
        # image = image.unsqueeze(0)
        
        
        
        # image = get_tensor()(Image.open(path).convert('RGB').resize((112,112))).unsqueeze(0)
        return image


def compute_features(files,mask_files, model,other_model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1,data_name="celeba",number_of_images=None,coordinates_path=None):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)
    # breakpoint()
    
    # breakpoint()
    
    dataset = MaskedImagePathDataset(files,maskfiles= mask_files, transforms=TF.ToTensor(),data_name=data_name,coordinate_path=coordinates_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)
    


    pred_arr = np.empty((len(files), 512))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device).squeeze(1)

        with torch.no_grad():
   
            pred = model(batch)
            

        pred = pred.cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr



def compute_features_wrapp(path,mask_path, IDLoss_model,Other_model, batch_size, dims, device,
                               num_workers=1,data_name="celeba", lables_file=None,args=None, number_of_images=None,coordinates_path=None):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        
        files = natsorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        # breakpoint()
        mask_path = pathlib.Path(mask_path)
        mask_files = natsorted([file for ext in IMAGE_EXTENSIONS
                       for file in mask_path.glob('*.{}'.format(ext))])
        
        if number_of_images is not None:
            files = files[:number_of_images]
            mask_files = mask_files[:number_of_images]
            # breakpoint()
        
        # breakpoint()
        if lables_file is None:
            # Extract all numbers before the dot using regular expression
            # breakpoint()
            pattern = r'[_\/.-]'

            # Split the file path using the pattern
            parts = [re.split(pattern, str(file.name)) for file in files]
            # breakpoint()
            # Filter out non-numeric parts and convert to integers
            numbers =[[int(par) for par in part if par.isdigit()] for part in parts]
            
            numbers= [ num[0] for num in numbers if len(num)>0]
            # breakpoint()
            mi_num= min(numbers)
            # if numbers[0]>28000: # CelebA-HQ Test my split #check 28000-29000: target 29000-30000: source
            numbers = [(num - mi_num) for num in numbers] # celeb

            # print(files)
            # print(numbers)
        
        else:
            numbers = []
            # Extract the clip name from the path (the folder before model_outputs)
            
            if args.results_subfolder is None or args.results_subfolder == "":
                clip_name = os.path.basename(path)
            else:
                clip_name = os.path.basename(os.path.dirname(path))
            # Read the labels from the file
            with open(lables_file, 'r') as f:
                lines = f.readlines()
                # breakpoint()
                for line in lines:
                    if clip_name in line:
                        # breakpoint()
                        number_str = line.strip().split(': ')[-1].replace('.jpg','')
                        if number_str.isdigit():
                            numbers.append(int(number_str))
                            break
        # print(f'Numbers: {numbers}')
        
        pred_arr = compute_features(files,mask_files, IDLoss_model,Other_model, batch_size,
                                               dims, device, num_workers,data_name=data_name,number_of_images=args.number_of_images,coordinates_path=coordinates_path)

    return pred_arr,numbers


def calculate_id_given_paths(paths, batch_size, device, dims, num_workers=1,data_name="celeba",args=None):
    """Calculates the FID of two paths"""
    if len(paths) != 4:
        raise RuntimeError('Invalid number of paths: %s' % len(paths))
    # get all subfolders in paths[1] and paths[3]
    subfolders1 = [os.path.basename(f.path) for f in os.scandir(paths[1]) if f.is_dir()]
    subfolders2 = [os.path.basename(f.path) for f in os.scandir(paths[3]) if f.is_dir()]
    # breakpoint()
    # assert len(subfolders1) == len(subfolders2), 'Number of subfolders in paths[1] and paths[3] must be equal'

    mean_simirities = []
    std_simirities = []
    mean_top1 = []
    mean_top5 = []
    vidds = []
    pathsdup= paths.copy()

    if args.arcface:
        IDLoss_model=IDLoss().cuda()
    
    video_names = []    
    with open(args.target_label_path, 'r') as f:
        lines = f.readlines()
        # breakpoint()
        for line in lines:
            vid = line.strip().split(': ')[0]
            video_names.append(vid)
                
    
    for subdirectory in video_names:
        

        paths = pathsdup.copy()
        vidname= subdirectory
        coordinates_path = os.path.join(args.crop_coordinates, subdirectory , vidname+"_inv_transforms.npy")
        
        if args.results_subfolder is None or args.results_subfolder == "":
            paths[1] = os.path.join(paths[1], subdirectory)
        else:
            paths[1] = os.path.join(paths[1], subdirectory, args.results_subfolder)
        if args.target_subfolder is None:
            paths[3] = os.path.join(paths[3], subdirectory)
        else:
            vidname= subdirectory
            paths[3] = os.path.join(paths[3], subdirectory, vidname+args.target_subfolder)

        # breakpoint()
        for p in paths:
            if not os.path.exists(p):
                raise RuntimeError('Invalid path: %s' % p)

        
    
        try:
            feat1,ori_lab = compute_features_wrapp(paths[0],paths[2], IDLoss_model,None, batch_size,
                                                dims, device, num_workers,data_name="celeba",args=args,coordinates_path=None,number_of_images=None)   #  celeb source face images
            feat2,swap_lab = compute_features_wrapp(paths[1],paths[3], IDLoss_model,None, batch_size,
                                                dims, device, num_workers,data_name=data_name, lables_file=args.target_label_path,args=args,number_of_images=args.number_of_images,coordinates_path=coordinates_path)
            
        except Exception as e:
            print(f"Error processing {subdirectory}: {e}")
            continue
        
        # breakpoint()
        # dot produc to get similarity
        # breakpoint()
        swap_lab = np.array(swap_lab)
        swap_lab= np.repeat(swap_lab, feat2.shape[0])
        dot_prod= np.dot(feat2,feat1.T)
        pred= np.argmax(dot_prod,axis=1)

        
        # find accuracy of top 1 and top 5
        top1 = np.sum(np.argmax(dot_prod,axis=1)==swap_lab)/len(swap_lab)
        top5_predictions = np.argsort(dot_prod, axis=1)[:, -5:]  # Get indices of top-5 predictions
        top5_correct = np.sum(np.any(top5_predictions == np.array(swap_lab)[:, np.newaxis], axis=1))
        top5 = top5_correct / len(swap_lab)  # Top-5 accuracy
        # top5 = np.sum(np.isin(np.argsort(dot_prod,axis=1)[:,-5:],swap_lab))/len(swap_lab)
    
        feat_sel=feat1[swap_lab]
        feat_sel=feat_sel/np.linalg.norm(feat_sel,axis=1,keepdims=True)
        feat2=feat2/np.linalg.norm(feat2,axis=1,keepdims=True)
        
        #VIDD
        diffs = feat2[:-1] - feat2[1:]  # Shape: (15, 512)
        vidd = np.linalg.norm(diffs, axis=1).mean()  # L2 norm along feature dim, then average
        # print("VIDD:", vidd.item())
        vidds.append(vidd)
        
        
        similarities=np.diagonal(np.dot(feat_sel,feat2.T))
        
        
        
        
        order=np.argsort(similarities)[::-1]
        value=np.sort(similarities)[::-1]
        # breakpoint()
        
    


        mean_simirities.append(np.mean(similarities))
        std_simirities.append(np.std(similarities))
        mean_top1.append(top1)
        mean_top5.append(top5)

      
    return mean_top1, mean_top5, mean_simirities, std_simirities,vidds


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = args.num_workers

    mean_top1, mean_top5, mean_similarities, std_similarities ,vidds = calculate_id_given_paths(args.path,
                                        args.batch_size,
                                        device,
                                        2048,
                                        num_workers,data_name=args.dataset,args=args)
    
    
    print('Top-1 accuracy: {:.2f}%'.format(np.mean(mean_top1) * 100))
    print('Top-5 accuracy: {:.2f}%'.format(np.mean(mean_top5) * 100))
    print('Mean ID feat:  {:.2f}'.format(np.mean(mean_similarities)))
    print('Std ID feat:  {:.2f}'.format(np.mean(std_similarities)))
    print('VIDD:  {:.2f}'.format(np.mean(vidds)))
    
    
    # if args.print_sim:
    #     print('Similarities: \n ')
    #     for i in range(len(similarities)):
    #         print(i,":",similarities[i])

if __name__ == '__main__':
    main()