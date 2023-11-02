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
from PIL import Image
import re
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from src.Face_models.encoders.model_irse import Backbone
# import clip
import torchvision
from eval_tool.ID_retrieval.iresnet50 import iresnet50,iresnet100
from eval_tool.ID_retrieval import cosface_net as cosface
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x
import cv2
import albumentations as A

# from inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=20,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--mask', type=bool, default=True,
                    help='whether to use mask or not')
# parser.add_argument('--dims', type=int, default=2048,
#                     choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
#                     help=('Dimensionality of Inception features to use. '
#                           'By default, uses pool3 features'))
parser.add_argument('path', type=str, nargs=4,
                    default=['/home/sanoojan/Paint_for_swap/dataset/FaceData/CelebAMask-HQ/CelebA-HQ-img', 'results/test_bench/results','dataset/FaceData/CelebAMask-HQ/src_mask','dataset/FaceData/CelebAMask-HQ/target_mask'],
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms
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
        image = get_tensor()(Image.open(path).convert('RGB').resize((112,112))).unsqueeze(0)
        return image


class MaskedImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files,maskfiles=None, transforms=None):
        self.files = files
        self.maskfiles = maskfiles  
        self.transforms = transforms
        self.trans=A.Compose([
            A.Resize(height=112,width=112)])
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
        image=cv2.imread(str(path))
        # ref_img = Image.open(ref_img_path).convert('RGB').resize((224,224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        mask_path = self.maskfiles[i]
        ref_mask_img = Image.open(mask_path).convert('L')
        ref_mask_img = np.array(ref_mask_img)  # Convert the label to a NumPy array if it's not already

        # preserve = [1,2,4,5,8,9 ,6,7,10,11,12 ] # CelebA-HQ
        preserve = [1,2,3,5,6,7,9]  # FFHQ or FF++
        
        # preserve = [1,2,4,5,8,9 ]
        ref_mask= np.isin(ref_mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        ref_converted_mask = np.zeros_like(ref_mask_img)
        ref_converted_mask[ref_mask] = 255
        ref_converted_mask=Image.fromarray(ref_converted_mask).convert('L')
        # convert to PIL image
        
        
        ref_mask_img_r = ref_converted_mask.resize(image.shape[1::-1], Image.NEAREST)
        ref_mask_img_r = np.array(ref_mask_img_r)
        image[ref_mask_img_r==0]=0
        
        image=self.trans(image=image)
        image=Image.fromarray(image["image"])
        image=get_tensor()(image)
        
        
        # ref_img=Image.fromarray(ref_img)
        
        # ref_img=get_tensor_clip()(ref_img)
        image = image.unsqueeze(0)
        
        
        
        # image = get_tensor()(Image.open(path).convert('RGB').resize((112,112))).unsqueeze(0)
        return image


def compute_features(files,mask_files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
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

    dataset = MaskedImagePathDataset(files,maskfiles=mask_files, transforms=TF.ToTensor())
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)
    


    pred_arr = np.empty((len(files), 512))

    start_idx = 0
    
    face_pool_1 = torch.nn.AdaptiveAvgPool2d((256, 256))
    face_pool_2 = torch.nn.AdaptiveAvgPool2d((112, 112))
    for batch in tqdm(dataloader):
        batch = batch.to(device).squeeze(1)

        with torch.no_grad():
            # x = face_pool_1(batch)  if batch.shape[2]!=256 else  batch # (1) resize to 256 if needed
            # x = x[:, :, 35:223, 32:220]  # (2) Crop interesting region
            # x = face_pool_2(x) # (3) resize to 112 to fit pre-trained model
            # breakpoint()
            pred = model(batch )
            # pred = model(batch )[0] for arcface
            
        # breakpoint()
        # # If model output is not scalar, apply global spatial average pooling.
        # # This happens if you choose a dimensionality not equal 2048.
        # if pred.size(2) != 1 or pred.size(3) != 1:
        #     pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        # pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        # print(pred.shape)
        pred = pred.cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_features_wrapp(path,mask_path, model, batch_size, dims, device,
                               num_workers=1):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        # breakpoint()
        mask_path = pathlib.Path(mask_path)
        mask_files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in mask_path.glob('*.{}'.format(ext))])
        # Extract all numbers before the dot using regular expression
        # breakpoint()
        pattern = r'[_\/.-]'

        # Split the file path using the pattern
        parts = [re.split(pattern, str(file.name)) for file in files]
        # breakpoint()
        # Filter out non-numeric parts and convert to integers
        numbers =[[int(par) for par in part if par.isdigit()] for part in parts]
        
        numbers= [ num[0] for num in numbers if len(num)>0]
        min_num= min(numbers)
        # if numbers[0]>28000: # CelebA-HQ Test my split #check 28000-29000: target 29000-30000: source
        numbers = [num-min_num for num in numbers] # celeb
        # breakpoint()
        pred_arr = compute_features(files,mask_files, model, batch_size,
                                               dims, device, num_workers)

    return pred_arr,numbers


def calculate_id_given_paths(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    
    # cosface_state_dict = torch.load("eval_tool/Face_rec_models/backbone.pth")
    # CosFace = iresnet50()
    
    # cosface_state_dict = torch.load("eval_tool/Face_rec_models/iresnet100_cosface/backbone.pth")
    # CosFace = iresnet100()
    
    
    cosface_state_dict = torch.load('/home/sanoojan/Paint_for_swap/eval_tool/Face_rec_models/cosface/net_sphere20_data_vggface2_acc_9955.pth')
    CosFace = cosface.sphere().cuda()
    
    
    CosFace.load_state_dict(cosface_state_dict)
    CosFace.eval()
    CosFace.to(device)
    
    


    # # model = InceptionV3([block_idx]).to(device)
    # CosFace  = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
    # path="/home/sanoojan/e4s/pretrained_ckpts/auxiliray/model_ir_se50.pth"
    # # CosFace.load_state_dict(torch.load('eval_tool/Face_rec_models/backbone.pth'))
    # CosFace.load_state_dict(torch.load(path))
    # CosFace.eval()
    # CosFace.to(device)
    

    feat1,ori_lab = compute_features_wrapp(paths[0],paths[2], CosFace, batch_size,
                                        dims, device, num_workers)
    feat2,swap_lab = compute_features_wrapp(paths[1],paths[3], CosFace, batch_size,
                                        dims, device, num_workers)
    # dot produc to get similarity
    # breakpoint()
    dot_prod= np.dot(feat2,feat1.T)
    pred= np.argmax(dot_prod,axis=1)
    # find accuracy of top 1 and top 5
    top1 = np.sum(np.argmax(dot_prod,axis=1)==swap_lab)/len(swap_lab)
    
    top5_predictions = np.argsort(dot_prod, axis=1)[:, -5:]  # Get indices of top-5 predictions
    top5_correct = np.sum(np.any(top5_predictions == np.array(swap_lab)[:, np.newaxis], axis=1))
    top5 = top5_correct / len(swap_lab)  # Top-5 accuracy
    # breakpoint()
    # top5 = np.sum(np.isin(np.argsort(dot_prod,axis=1)[:,-5:],swap_lab))/len(swap_lab)
    feat_sel=feat1[swap_lab]
    feat_sel=feat_sel/np.linalg.norm(feat_sel,axis=1,keepdims=True)
    feat2=feat2/np.linalg.norm(feat2,axis=1,keepdims=True)
    similarities=np.diagonal(np.dot(feat_sel,feat2.T))
    #print from highest to lowest with index
    
    
    
    order=np.argsort(similarities)[::-1]
    value=np.sort(similarities)[::-1]
    # breakpoint()
    
    Mean_dot_prod= np.mean(similarities)
    

    # breakpoint()
    return top1,top5,Mean_dot_prod


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

    top1,top5,Mean_dot_prod= calculate_id_given_paths(args.path,
                                          args.batch_size,
                                          device,
                                          2048,
                                          num_workers)
    print('Top-1 accuracy: {:.2f}%'.format(top1 * 100))
    print('Top-5 accuracy: {:.2f}%'.format(top5 * 100))
    print('Mean ID feat:  {:.2f}'.format(Mean_dot_prod))

if __name__ == '__main__':
    main()