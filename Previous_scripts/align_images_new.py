
    
##############################################################################
############# Align comparing different methods #############################

Simswap_path="/home/sanoojan/other_swappers/SimSwap/output/CelebA/results"
e4s_path="/home/sanoojan/e4s/Results/testbench/results_Original_ckpt_without_crop/results"
DiffFace_path="/home/sanoojan/other_swappers/DiffFace/results/CelebA/results"
DiffFace_10_path="/home/sanoojan/other_swappers/DiffFace/results/Celeba_20/results"

Diffswap_path="/home/sanoojan/other_swappers/DiffSwap/all_images_with_folders_named_2_celeba"
Diffswap_10_path="/home/sanoojan/other_swappers/DiffSwap/all_images_with_folders_named_2_celeba_20"
HifiFace_path="/home/sanoojan/other_swappers/hififace/celeba_results"
MegaFs_path="/home/sanoojan/other_swappers/MegaFs/Celeba_outs"
FaceDancer_path="/home/sanoojan/other_swappers/FaceDancer/FaceDancer_c_HQ/results"
# ours_path="results_grad/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16_with_src_hair/results"
# ours_path="results_FINALS/v5_Two_CLIP_proj_with_multiple_ID_losses_ep17_3_75/results"
ours_path="results_FINALS/v5_Two_CLIP_proj_154_ep_last/results"
ours_20_path="results_FINALS/v5_Two_CLIP_proj_154_ep_3.5_20/results"
ours_5_path="results_FINALS/v5_Two_CLIP_proj_154_ep_3.5_5/results"
src_path="dataset/FaceData/CelebAMask-HQ/Val"
target_path="dataset/FaceData/CelebAMask-HQ/Val_target"
save_path="Aligned/Aligne_CelebA_demo_eff.png"
# select_images=[3,7,25,31,32,64,69,81,82,88,102]
# select_images=[7,25,32,64,81,82,88,102]
# select_images=[7,25,541,64,81,82,556,570]
# select_images=[25,541,64,81,570]   # crop half 
# select_images= [82,570,489]

# select_images=[25,541,64,81,570,50,104,166]    # checking more pose variations
# select_images=[6,53,57,60,78,174,141]
# select_images=[25,541,570,141,104]
# # select_images=[129 , 64 ,106 ,115, 174 ,200 ,201 ,81 ,290,50 ,6] # This is for supplementary
select_images=[6,407]

# select_images=[7,24,32,64,69,81]
# select_images=[134,146,161,165,166,172]
# select_images=[3,7,25,31,32]
# select_images= [64,69,81,82,88,102]

path_list=[src_path,target_path,ours_5_path,ours_path,Diffswap_10_path,Diffswap_path,DiffFace_10_path,DiffFace_path]

Labels=["Source","Target","Ours_5","Ours_20","Ours","DiffSwap_10","DiffFace_10","DiffSwap","DiffFace"]

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import natsort
import re

def read_images(path_list, select_images):
    images = []
    for path in path_list:
        filenames = os.listdir(path)
        filenames = natsort.natsorted(filenames)  # Sort filenames naturally
        
        pattern = r'[_\/.-]'

        # Split the file path using the pattern
        parts = [re.split(pattern, str(file)) for file in filenames]
        # breakpoint()
        # Filter out non-numeric parts and convert to integers
        numbers =[[int(par) for par in part if par.isdigit()] for part in parts]
        
        numbers= [ num[0] for num in numbers if len(num)>0]
        
        mi_num= min(numbers)
        numbers = [(num - mi_num) for num in numbers] # celeb
        # find the index of the select images
        select_images2 = [numbers.index(num) for num in select_images]
        # breakpoint()
        selected_filenames = [filenames[i] for i in select_images2]
        
        images.append([cv2.cvtColor(cv2.imread(os.path.join(path, filename)), cv2.COLOR_BGR2RGB) for filename in selected_filenames])
    return images

def visualize_images(images, save_path):
    num_paths = len(images)
    num_images_per_path = len(images[0])
    x_size = num_paths * 5
    y_size = num_images_per_path * 5
    fig, ax = plt.subplots(num_images_per_path, num_paths, figsize=(x_size, y_size))

    for i in range(num_images_per_path):
        for j in range(num_paths):
            ax[i, j].imshow(images[j][i])
            ax[i, j].axis('off')
            if i == 0:
                ax[i, j].set_title(Labels[j], fontsize=25)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    



# images = read_images(path_list, select_images)
# visualize_images(images,save_path)



Simswap_path="/home/sanoojan/other_swappers/SimSwap/output/FFHQ/results"
e4s_path="/home/sanoojan/e4s/Results/testbench/results_on_FFHQ_orig_ckpt/results"
DiffFace_path="/home/sanoojan/other_swappers/DiffFace/results/FFHQ/results"
Diffswap_path="/home/sanoojan/other_swappers/DiffSwap/all_images_with_folders_named_2_FFHQ"
HifiFace_path="/home/sanoojan/other_swappers/hififace/FFHQ_results"
FaceDancer_path="/home/sanoojan/other_swappers/FaceDancer/FaceDancer_c_HQ-FFHQ/results"
# ours_path="results_grad/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16_with_src_hair/results"
ours_path="/home/sanoojan/Face-in-Fusion/results/FFHQ/REFace_fft_at_attn_final/results"
REFace_path="/home/sanoojan/Face-in-Fusion/results/FFHQ/REFace/results"
save_path="Aligned/Aligne_FFHQ_new.png"


src_path="dataset/FaceData/FFHQ/Val"
target_path="dataset/FaceData/FFHQ/Val_target"

# select_images=[3,7,25,31,32,64,69,81,82,88,102]
# select_images=[3,5,9,11,18,117,155,159]
select_images=[9,11,18,155,159]
select_images=[9,11,18,155,159,10,23,76,95,177,193]
select_images=[9,11,18,155,308]
select_images=[  4 ,  7, 101 ,193 ,203 ,213 ,177, 279 ,117 ,331 ,359] #supp
select_images=[686,792,19,960,300,303,310,329,374,375] # extreme cond

select_images=[9,11,18,155,159]
select_images=[7,18,29,36,38,49,51,95]
select_images=[7,18,29,36,49,51,95]



# select_images=[100,101,118,128,140,312,330]

# select_images=[134,146,161,165,166,172]
# select_images=[3,7,25,31,32]
# select_images= [64,69,81,82,88,102]


path_list=[src_path,target_path,Simswap_path,e4s_path,DiffFace_path,Diffswap_path,REFace_path,ours_path]

Labels=["Source","Target","SimSwap","E4S","DiffFace","DiffSwap","REFace","Ours"]

images = read_images(path_list, select_images)
visualize_images(images,save_path)



############################Head Swap #########################

# Hair Swap Results 

# Ours_no_grad_path="results/v4_reconstruct_img_train_2_step_ep_38/results"
# Ours_grad_path="results_grad/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16_with_src_hair/results"

# # ours_path="results_grad/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16_with_src_hair/results"
# # ours_path="results_grad/50_v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_noclip_same_image_ep20/results"
# Ours_old_path="results_FINAL/v5_Two_CLIP_proj_with_multiple_ID_losses_ep12_75_hair_swap/results"
# Ours_path="results_FINAL/v5_SRC_CLIP_proj_with_multiple_ID_losses_random_masks_hair_swap_3_75/results"
# Ours_final_path="results_FINAL/v5_Two_CLIP_proj_with_multiple_ID_losses_random_mask_ep_16_swap_hair/results"
# src_path="dataset/FaceData/CelebAMask-HQ/Val"
# target_path="dataset/FaceData/CelebAMask-HQ/Val_target"
# Mask_path="results_FINAL/v5_SRC_CLIP_proj_with_multiple_ID_losses_random_masks_hair_swap_3_75/samples"


# select_images=[61,81,82,83,111,163,225]



# path_list=[src_path,target_path,Mask_path,Ours_old_path,Ours_final_path]

# Labels=["Source","Target","Inpaint","Ours w/o\nmask shuffling","Ours"]

# import os
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import natsort
# import re

# def read_images(path_list, select_images):
#     images = []
#     for path in path_list:
#         filenames = os.listdir(path)
#         # filter paths 
#         if "samples" in path:
#             filenames = [filename for filename in filenames if "inpaint" in filename]
        
#         filenames = natsort.natsorted(filenames)  # Sort filenames naturally
        
        
        
#         selected_filenames = [filenames[i] for i in select_images]
#         images.append([cv2.cvtColor(cv2.imread(os.path.join(path, filename)), cv2.COLOR_BGR2RGB) for filename in selected_filenames])
#     return images

# def visualize_images(images):
#     num_paths = len(images)
#     num_images_per_path = len(images[0])
#     x_size=num_paths*6
#     y_size=num_images_per_path*6
#     fig, ax = plt.subplots(num_images_per_path, num_paths,figsize=(x_size, y_size))

#     for i in range(num_images_per_path):
#         for j in range(num_paths):
#             ax[i, j].imshow(images[j][i])
#             ax[i, j].axis('off')
#             if i == 0:
#                 ax[i, j].set_title(Labels[j], fontsize=40)
                
#     plt.subplots_adjust(wspace=0, hspace=0)
#     # save plot
#     plt.savefig("Image_outputs/visualize_hair_swap.png")
    
#     # plt.show()

# # images = read_images(path_list, select_images)
# # # save 


# # visualize_images(images)


################################# compare clip dissentangle #####################


# # Simswap_path="/home/sanoojan/other_swappers/SimSwap/output/CelebA/results"
# # e4s_path="/home/sanoojan/e4s/Results/testbench/results_Original_ckpt_without_crop/results"
# # DiffFace_path="/home/sanoojan/other_swappers/DiffFace/results/CelebA/results"
# # # ours_path="results_grad/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16_with_src_hair/results"
# ours_path="results_FFHQ_FINAL/v5_Two_CLIP_proj_with_multiple_ID_losses_random_mask_hair_FINAL_ep14_HairSwap_sc_3_75/results"

# source_path="dataset/FaceData/CelebAMask-HQ/Val"
# target_path="dataset/FaceData/CelebAMask-HQ/Val_target"
# Nose_only_path="results_FINAL/v5_Two_CLIP_proj_with_multiple_ID_losses_ep13_75_nose_only/results"
# Nose_and_mouth="results_FINAL/v5_Two_CLIP_proj_with_multiple_ID_losses_ep13_75_nose_and_mouth_only/results"
# Nose_and_mouth_and_eyes="results_FINAL/v5_Two_CLIP_proj_with_multiple_ID_losses_ep13_75_eyes_nose_and_mouth_only/results"
# Full_face="results_FINAL/v5_Two_CLIP_proj_with_multiple_ID_losses_ep13_75/results"
# # select_images=[3,7,25,31,32,64,69,81,82,88,102]
# # select_images=[14,24]
# # select_images=[134,146,161,165,166,172]
# # select_images=[3,7,25,31,32]
# # select_images= [64,69,81,82,88,102]
# select_images=[13,30,50,62,36,63,88]

# path_list=[source_path,target_path,Nose_only_path,Nose_and_mouth,Nose_and_mouth_and_eyes,Full_face]

# Labels=["Source","Target","Nose_only","Nose_and_mouth","Nose_and_mouth_and_eyes","Full_face"]

# import os
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import natsort

# def read_images(path_list, select_images):
#     images = []
#     for path in path_list:
#         filenames = os.listdir(path)
#         filenames = natsort.natsorted(filenames)  # Sort filenames naturally
#         selected_filenames = [filenames[i] for i in select_images]
#         images.append([cv2.cvtColor(cv2.imread(os.path.join(path, filename)), cv2.COLOR_BGR2RGB) for filename in selected_filenames])
#     return images

# def visualize_images(images):
#     num_paths = len(images)
#     num_images_per_path = len(images[0])
#     x_size=num_paths*5
#     y_size=num_images_per_path*5
#     fig, ax = plt.subplots(num_images_per_path, num_paths,figsize=(x_size, y_size))

#     for i in range(num_images_per_path):
#         for j in range(num_paths):
#             ax[i, j].imshow(images[j][i])
#             ax[i, j].axis('off')
#             if i == 0:
#                 ax[i, j].set_title(Labels[j], fontsize=20)
#     plt.subplots_adjust(wspace=0, hspace=0)
    
#     plt.subplots_adjust(wspace=0, hspace=0)
#     # save plot
#     plt.savefig("Image_outputs/clip_dissentanglement.png", bbox_inches='tight', pad_inches=0)
#     # plt.show()

# # images = read_images(path_list, select_images)
# # visualize_images(images)