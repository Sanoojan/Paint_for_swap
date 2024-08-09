from src.utils.alignmengt import crop_faces, calc_alignment_coefficients, crop_faces_from_image
from PIL import Image
import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps
from tqdm import tqdm


import dlib
import cv2
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def crop_and_align_face(target_files):
    image_size = 1024
    scale = 1.0
    center_sigma = 0
    xy_sigma = 0
    use_fa = False
    
    print('Aligning images')
    crops, orig_images, quads = crop_faces(image_size, target_files, scale, center_sigma=center_sigma, xy_sigma=xy_sigma, use_fa=use_fa)
    
    inv_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads
    ]
    
    return crops, orig_images, quads, inv_transforms

def crop_and_align_face_img(frame):
    image_size = 1024
    scale = 1.0
    center_sigma = 0
    xy_sigma = 0
    use_fa = False
    
    print('Aligning images')
    crops, orig_images, quads = crop_faces_from_image(image_size, frame, scale, center_sigma=center_sigma, xy_sigma=xy_sigma, use_fa=use_fa)
    
    inv_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads
    ]
    
    return crops, orig_images, quads, inv_transforms

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a photograph of an astronaut riding a horse",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="results_video/debug"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
        default="True"
    )
    parser.add_argument(
        "--Start_from_target",
        action='store_true',
        help="if enabled, uses the noised target image as the starting ",
    )
    parser.add_argument(
        "--only_target_crop",
        action='store_true',
        help="if enabled, uses the noised target image as the starting ",
        default=True
    )
    parser.add_argument(
        "--target_start_noise_t",
        type=int,
        default=1000,
        help="target_start_noise_t",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--target_video",
        type=str,
        help="target_video",
        default="examples/faceswap/Andy2.mp4",
    )
    parser.add_argument(
        "--src_image",
        type=str,
        help="src_image",
        default="/share/data/drive_3/Sanoojan/needed/Paint_for_swap/examples/faceswap/source.jpg"
    )
    parser.add_argument(
        "--src_image_mask",
        type=str,
        help="src_image_mask",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/debug.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/Paint-by-Example/ID_Landmark_CLIP_reconstruct_img_train/PBE/celebA/2023-10-07T21-09-06_v4_reconstruct_img_train/checkpoints/last.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    
    parser.add_argument('--faceParser_name', default='default', type=str, help='face parser name, [ default | segnext] is currently supported.')
    parser.add_argument('--faceParsing_ckpt', type=str, default="pretrained/face_parsing/79999_iter.pth")  
    parser.add_argument('--segnext_config', default='', type=str, help='Path to pre-trained SegNeXt faceParser configuration file, '
                                                                        'this option is valid when --faceParsing_ckpt=segenext')
            
    parser.add_argument('--save_vis', action='store_true')
    parser.add_argument('--seg12',default=True, action='store_true')
    
    opt = parser.parse_args()


    faceParsing_model = init_faceParsing_pretrained_model(opt.faceParser_name, opt.faceParsing_ckpt, opt.segnext_config)
        

    Image_path_='dataset/FaceData/CelebAMask-HQ/Val_target'
    second_path_='/home/sanoojan/e4s/Results/testbench/reenact/results'
    mask_path_='dataset/FaceData/CelebAMask-HQ/target_mask'
    save_path_='/share/data/drive_3/Sanoojan/needed/Paint_for_swap/intermediate_renact/results'
    # get image list
    # image_list = glob.glob(os.path.join(Image_path, '*.png'))

    im_start=28000
    sec_start=29000
    
    for i in tqdm(range(0,1000)):
        image_path=os.path.join(Image_path_,str(im_start+i)+'.jpg')
        mask_path=os.path.join(mask_path_,str(im_start+i)+'.png')
        second_image_path=os.path.join(second_path_,str(sec_start+i)+'.png')
        save_path=os.path.join(save_path_,str(im_start+i)+'.jpg')
        
        preserve=[1,2,4,5,8,9 ,6,7,10,11,12 ]
        preserve2=[1,2,3,5,6,7,9]
        mask_img = Image.open(mask_path).convert('L')
        mask_img = np.array(mask_img)  # Convert the label to a NumPy array if it's not already
        mask = np.isin(mask_img, preserve)

        converted_mask = np.zeros_like(mask_img)
        converted_mask[mask] = 255
     


        image = cv2.imread(image_path)

        #reshape to 25
        image = cv2.resize(image, (512, 512))

        second_image = cv2.imread(second_image_path)
        pil_img = Image.fromarray(second_image)
        mask_scond = faceParsing_demo(faceParsing_model, pil_img, convert_to_seg12=opt.seg12, model_name=opt.faceParser_name)
        mask_img_2 = np.array(mask_scond)  # Convert the label to a NumPy array if it's not already
        mask_2 = np.isin(mask_img_2, preserve2)

        converted_mask_2 = np.zeros_like(mask_img_2)
        converted_mask_2[mask_2] = 255
        
        second_image = cv2.resize(second_image, (512, 512))
        
        # make noise of the same size as the image
        noise = np.random.randint(0, 256, image.shape).astype('uint8')
        
        # second_image[converted_mask_2 == 0] = noise[converted_mask_2 == 0]
        
        
        # replace the image with the second image where the mask is 255
        second_image[converted_mask == 0] = image[converted_mask == 0]

        #

        
        #save the second image
        cv2.imwrite(save_path,second_image)
        
        # display mask_img
        # plt.imshow(mask_img, cmap='gray')

        # # Display the image with landmarks
        # plt.imshow(cv2.cvtColor(second_image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()
        
        
if __name__ == "__main__":
    main()