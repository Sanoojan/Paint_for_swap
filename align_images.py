####Stack images horizontally ####

# import os
# import cv2
# import numpy as np

# # Path to the directory containing images
# image_dir = "examples/FaceSwap_10/Rebuttal_check_large_black_without_neck/results/results/5"
# target_image_path="examples/FaceSwap_10/Rebuttal_check_large_black_without_neck/results/5"

# # Indices of images to stack
# # image_indices = [0, 1, 4,7,13,14,15,16,17]
# # image_indices = [19,20,21,22,23,28,29,30,31,34]
# image_indices=[13,16,22,29,31,30]

# # List all files in the directory
# all_files = os.listdir(image_dir)

# # Filter out image files (assuming PNG format)
# image_files = [f for f in all_files if f.endswith('.png')]

# # Sort image files to ensure consistent ordering
# image_files.sort()

# # Load the images based on the sorted file list and provided indices
# images = []
# for idx in image_indices:
#     if idx < len(image_files):
#         image_path = os.path.join(image_dir, image_files[idx])
#         image = cv2.imread(image_path)
#         if image is not None:
#             images.append(image)
#         else:
#             print(f"Warning: Image at {image_path} could not be loaded.")
#     else:
#         print(f"Warning: Index {idx} is out of range for the available images.")

# # Stack images horizontally
# if images:
#     stacked_image = np.hstack(images)
    
#     # Save the resulting image
    
#     # Save the resulting image
#     output_path = "Aligned/Aligned_ours_final.png"
#     cv2.imwrite(output_path, stacked_image)
#     print(f"Stacked image saved as {output_path}")
# else:
#     print("No images were loaded. Please check the image paths and indices.")


############################################################################################################
#####Bottom corner add images #######

# import os
# import cv2
# import numpy as np

# # Paths to the directories containing images and ground truth images
# image_dir = "examples/FaceSwap_10/Rebuttal_check_large_black_without_neck/results/results/5"
# target_image_path = "examples/FaceSwap_10/Rebuttal_check_large_black_without_neck/results/5"

# # Indices of images to stack
# image_indices = [13, 16, 22, 29, 31, 30]

# # List all files in the directory
# all_files = os.listdir(image_dir)

# # Filter out image files (assuming PNG format)
# image_files = [f for f in all_files if f.endswith('.png')]

# # Sort image files to ensure consistent ordering
# image_files.sort()

# # Function to overlay ground truth image onto main image
# def overlay_ground_truth(main_image, gt_image_path):
#     gt_image = cv2.imread(gt_image_path)
#     if gt_image is not None:
#         # Resize ground truth image to fit in the bottom right corner
#         h_main, w_main = main_image.shape[:2]
#         h_gt, w_gt = gt_image.shape[:2]
        
#         scale_factor = min(w_main / w_gt, h_main / h_gt) * 0.3  # Scale down to 25% of main image dimensions
#         new_size = (int(w_gt * scale_factor), int(h_gt * scale_factor))
#         gt_image_resized = cv2.resize(gt_image, new_size, interpolation=cv2.INTER_AREA)
        
#         # Position ground truth image in the bottom right corner
#         h_gt_resized, w_gt_resized = gt_image_resized.shape[:2]
#         x_offset = w_main - w_gt_resized
#         y_offset = h_main - h_gt_resized
        
#         main_image[y_offset:y_offset+h_gt_resized, x_offset:x_offset+w_gt_resized] = gt_image_resized
#     else:
#         print(f"Warning: Ground truth image at {gt_image_path} could not be loaded.")
#     return main_image

# # Load the images based on the sorted file list and provided indices
# images = []
# for idx in image_indices:
#     if idx < len(image_files):
#         image_path = os.path.join(image_dir, image_files[idx])
#         image = cv2.imread(image_path)
#         if image is not None:
#             # Construct the ground truth image path
#             gt_image_path = os.path.join(target_image_path, f"{image_files[idx].split('.')[0]}_GT.png")
#             # Overlay the ground truth image onto the main image
#             image_with_gt = overlay_ground_truth(image, gt_image_path)
#             images.append(image_with_gt)
#         else:
#             print(f"Warning: Image at {image_path} could not be loaded.")
#     else:
#         print(f"Warning: Index {idx} is out of range for the available images.")

# # Stack images horizontally
# if images:
#     stacked_image = np.hstack(images)
    
#     # Save the resulting image
#     output_path = "Aligned/Aligned_ours_final_2.png"
#     cv2.imwrite(output_path, stacked_image)
#     print(f"Stacked image saved as {output_path}")
# else:
#     print("No images were loaded. Please check the image paths and indices.")

############################################################################################################

###### Left right corner add images ######
# import os
# import cv2
# import numpy as np

# # Paths to the directories containing images and ground truth images
# image_dir = "examples/FaceSwap_10/Rebuttal_check_large_black_without_neck/results/results/5"
# target_image_path = "examples/FaceSwap_10/Rebuttal_check_large_black_without_neck/results/5"
# source_image_path = "examples/FaceSwap_10/Rebuttal_check_large_black_without_neck/Outs/source_cropped/1.png"

# # Indices of images to stack
# image_indices = [13, 16, 22, 29, 31, 35]


# # List all files in the directory
# all_files = os.listdir(image_dir)

# # Filter out image files (assuming PNG format)
# image_files = [f for f in all_files if f.endswith('.png')]

# # Sort image files to ensure consistent ordering
# image_files.sort()

# # Load the source image once
# source_image = cv2.imread(source_image_path)
# if source_image is None:
#     raise FileNotFoundError(f"Source image at {source_image_path} could not be loaded.")

# # Function to overlay ground truth and source images onto main image
# def overlay_images(main_image, gt_image_path, source_image):
#     # Resize ground truth image to fit in the bottom right corner
#     gt_image = cv2.imread(gt_image_path)
#     if gt_image is not None:
#         h_main, w_main = main_image.shape[:2]
#         h_gt, w_gt = gt_image.shape[:2]
        
#         scale_factor_gt = min(w_main / w_gt, h_main / h_gt) * 0.25  # Scale down to 25% of main image dimensions
#         new_size_gt = (int(w_gt * scale_factor_gt), int(h_gt * scale_factor_gt))
#         gt_image_resized = cv2.resize(gt_image, new_size_gt, interpolation=cv2.INTER_AREA)
        
#         # Position ground truth image in the bottom right corner
#         h_gt_resized, w_gt_resized = gt_image_resized.shape[:2]
#         x_offset_gt = w_main - w_gt_resized
#         y_offset_gt = h_main - h_gt_resized
        
#         main_image[y_offset_gt:y_offset_gt+h_gt_resized, x_offset_gt:x_offset_gt+w_gt_resized] = gt_image_resized

#         # Add "T" text to ground truth image
#         cv2.putText(main_image, 'T', (x_offset_gt + 10, y_offset_gt + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#     else:
#         print(f"Warning: Ground truth image at {gt_image_path} could not be loaded.")
    
#     # Resize source image to fit in the bottom left corner
#     h_source, w_source = source_image.shape[:2]
#     scale_factor_source = min(w_main / w_source, h_main / h_source) * 0.25  # Scale down to 25% of main image dimensions
#     new_size_source = (int(w_source * scale_factor_source), int(h_source * scale_factor_source))
#     source_image_resized = cv2.resize(source_image, new_size_source, interpolation=cv2.INTER_AREA)
    
#     # Position source image in the bottom left corner
#     h_source_resized, w_source_resized = source_image_resized.shape[:2]
#     x_offset_source = 0
#     y_offset_source = h_main - h_source_resized
    
#     main_image[y_offset_source:y_offset_source+h_source_resized, x_offset_source:x_offset_source+w_source_resized] = source_image_resized

#     # Add "S" text to source image
#     cv2.putText(main_image, 'S', (x_offset_source + 10, y_offset_source + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
#     return main_image

# # Load the images based on the sorted file list and provided indices
# images = []
# for idx in image_indices:
#     if idx < len(image_files):
#         image_path = os.path.join(image_dir, image_files[idx])
#         image = cv2.imread(image_path)
#         if image is not None:
#             # Construct the ground truth image path
#             gt_image_path = os.path.join(target_image_path, f"{image_files[idx].split('.')[0]}_GT.png")
#             # Overlay the ground truth and source images onto the main image
#             image_with_overlays = overlay_images(image, gt_image_path, source_image)
#             images.append(image_with_overlays)
#         else:
#             print(f"Warning: Image at {image_path} could not be loaded.")
#     else:
#         print(f"Warning: Index {idx} is out of range for the available images.")

# # Stack images horizontally
# if images:
#     stacked_image = np.hstack(images)
    
#     # Save the resulting image
#     output_path = "Aligned/Aligned_ours_final_4.png"
#     cv2.imwrite(output_path, stacked_image)
#     print(f"Stacked image saved as {output_path}")
# else:
#     print("No images were loaded. Please check the image paths and indices.")
############################################################################################################


#########################################################################################################

##### Left right corner add images ######
# import os
# import cv2
# import numpy as np

# # Paths to the fixed target image and source images
# target_image_path = "examples/FaceSwap_10/Rebuttal_check_few_target_without_neck/Outs/target_cropped/1.png"
# source_image_global_path = "examples/FaceSwap_10/Rebuttal_check_few_target_without_neck/Outs/source_cropped"
# output_global_path = "examples/FaceSwap_10/Rebuttal_check_few_target_without_neck/results/results"
# # source_image_global_path="examples/FaceSwap_10/Rebuttal_check_large_black_without_neck/Outs/target_cropped"
# # IDs for source images and corresponding output image IDs
# # source_image_ids = [9,34,18,23,2,26,14]
# # corresponding_output_ids = [2,8,12,17,21,30,32]
# source_image_ids = [9,34,18,2,26,14]
# corresponding_output_ids = [2,8,12,21,30,32]

# # Target ID for output images
# target_id = 1

# # Load the fixed target image
# target_image = cv2.imread(target_image_path)
# if target_image is None:
#     raise FileNotFoundError(f"Target image at {target_image_path} could not be loaded.")

# # Function to overlay source and target images onto the output image
# def overlay_images(output_image, source_image_path, target_image):
#     # Resize source image to fit in the bottom left corner
#     source_image = cv2.imread(source_image_path)
#     if source_image is not None:
#         h_output, w_output = output_image.shape[:2]
#         h_source, w_source = source_image.shape[:2]
        
#         scale_factor_source = min(w_output / w_source, h_output / h_source) * 0.25  # Scale down to 25% of output image dimensions
#         new_size_source = (int(w_source * scale_factor_source), int(h_source * scale_factor_source))
#         source_image_resized = cv2.resize(source_image, new_size_source, interpolation=cv2.INTER_AREA)
        
#         # Position source image in the bottom left corner
#         h_source_resized, w_source_resized = source_image_resized.shape[:2]
#         x_offset_source = 0
#         y_offset_source = h_output - h_source_resized
        
#         output_image[y_offset_source:y_offset_source+h_source_resized, x_offset_source:x_offset_source+w_source_resized] = source_image_resized
        
#         # Add "S" text to source image
#         cv2.putText(output_image, 'S', (x_offset_source + 10, y_offset_source + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#     else:
#         print(f"Warning: Source image at {source_image_path} could not be loaded.")
    
#     # Resize target image to fit in the bottom right corner
#     h_target, w_target = target_image.shape[:2]
    
#     scale_factor_target = min(w_output / w_target, h_output / h_target) * 0.25  # Scale down to 25% of output image dimensions
#     new_size_target = (int(w_target * scale_factor_target), int(h_target * scale_factor_target))
#     target_image_resized = cv2.resize(target_image, new_size_target, interpolation=cv2.INTER_AREA)
    
#     # Position target image in the bottom right corner
#     h_target_resized, w_target_resized = target_image_resized.shape[:2]
#     x_offset_target = w_output - w_target_resized
#     y_offset_target = h_output - h_target_resized
    
#     output_image[y_offset_target:y_offset_target+h_target_resized, x_offset_target:x_offset_target+w_target_resized] = target_image_resized
    
#     # Add "T" text to target image
#     cv2.putText(output_image, 'T', (x_offset_target + 10, y_offset_target + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
#     return output_image

# # Load the images and process them
# images = []
# for source_id, output_id in zip(source_image_ids, corresponding_output_ids):
#     source_image_path = os.path.join(source_image_global_path, f"{source_id}.png")
#     output_image_path = os.path.join(output_global_path, f"{output_id}/00000000000{target_id}.png")
    
#     # Load the output image
#     output_image = cv2.imread(output_image_path)
#     if output_image is None:
#         print(f"Warning: Output image at {output_image_path} could not be loaded.")
#         continue
    
#     # Make a copy of the output image to avoid modifying the original
#     output_image_copy = output_image.copy()
    
#     # Overlay the source and target images onto the output image
#     image_with_overlays = overlay_images(output_image_copy, source_image_path, target_image)
#     images.append(image_with_overlays)

# # Stack images horizontally
# if images:
#     stacked_image = np.hstack(images)
    
#     # Save the resulting image
#     output_path = "Aligned/Aligned_ours_final_iron_man.png"
#     cv2.imwrite(output_path, stacked_image)
#     print(f"Stacked image saved as {output_path}")
# else:
#     print("No images were loaded. Please check the image paths and indices.")



############################################################################################################
# import os
# import cv2
# import numpy as np

# # Paths to the fixed target image and source images
# target_image_path = "examples/FaceSwap_10/Rebuttal_check_few_target_without_neck_3/Outs/target_cropped/0.png"
# source_image_global_path = "examples/FaceSwap_10/Rebuttal_check_few_target_without_neck_3/Outs/source_cropped"
# output_global_path = "examples/FaceSwap_10/Rebuttal_check_few_target_without_neck_3/results/results"

# # IDs for source images and corresponding output image IDs
# # source_image_ids = [14,9,27,1,8,15,2,23,5]
# # corresponding_output_ids = [33,32,24,23,21,19,17,16,15]

# source_image_ids = [14,9,1,8,2,23]
# corresponding_output_ids = [33,32,23,21,17,16]

# # Target ID for output images
# target_id = 0

# # Load the fixed target image
# target_image = cv2.imread(target_image_path)
# if target_image is None:
#     raise FileNotFoundError(f"Target image at {target_image_path} could not be loaded.")

# # Function to overlay source and target images onto the output image
# def overlay_images(output_image, source_image_path, target_image):
#     # Resize source image to fit in the bottom left corner
#     source_image = cv2.imread(source_image_path)
#     if source_image is not None:
#         h_output, w_output = output_image.shape[:2]
#         h_source, w_source = source_image.shape[:2]
        
#         scale_factor_source = min(w_output / w_source, h_output / h_source) * 0.25  # Scale down to 25% of output image dimensions
#         new_size_source = (int(w_source * scale_factor_source), int(h_source * scale_factor_source))
#         source_image_resized = cv2.resize(source_image, new_size_source, interpolation=cv2.INTER_LANCZOS4)
        
#         # Position source image in the bottom left corner
#         h_source_resized, w_source_resized = source_image_resized.shape[:2]
#         x_offset_source = 0
#         y_offset_source = h_output - h_source_resized
        
#         output_image[y_offset_source:y_offset_source+h_source_resized, x_offset_source:x_offset_source+w_source_resized] = source_image_resized
        
#         # Add "S" text to source image
#         cv2.putText(output_image, 'S', (x_offset_source + 10, y_offset_source + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
#     else:
#         print(f"Warning: Source image at {source_image_path} could not be loaded.")
    
#     # Resize target image to fit in the bottom right corner
#     h_target, w_target = target_image.shape[:2]
    
#     scale_factor_target = min(w_output / w_target, h_output / h_target) * 0.25  # Scale down to 25% of output image dimensions
#     new_size_target = (int(w_target * scale_factor_target), int(h_target * scale_factor_target))
#     target_image_resized = cv2.resize(target_image, new_size_target, interpolation=cv2.INTER_LANCZOS4)
    
#     # Position target image in the bottom right corner
#     h_target_resized, w_target_resized = target_image_resized.shape[:2]
#     x_offset_target = w_output - w_target_resized
#     y_offset_target = h_output - h_target_resized
    
#     output_image[y_offset_target:y_offset_target+h_target_resized, x_offset_target:x_offset_target+w_target_resized] = target_image_resized
    
#     # Add "T" text to target image
#     cv2.putText(output_image, 'T', (x_offset_target + 10, y_offset_target + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
    
#     return output_image

# # Load the images and process them
# images = []
# for source_id, output_id in zip(source_image_ids, corresponding_output_ids):
#     source_image_path = os.path.join(source_image_global_path, f"{source_id}.png")
#     output_image_path = os.path.join(output_global_path, f"{output_id}/00000000000{target_id}.png")
    
#     # Load the output image
#     output_image = cv2.imread(output_image_path)
#     if output_image is None:
#         print(f"Warning: Output image at {output_image_path} could not be loaded.")
#         continue
    
#     # Make a copy of the output image to avoid modifying the original
#     output_image_copy = output_image.copy()
    
#     # Overlay the source and target images onto the output image
#     image_with_overlays = overlay_images(output_image_copy, source_image_path, target_image)
#     images.append(image_with_overlays)

# # Stack images horizontally
# if images:
#     stacked_image = np.hstack(images)
    
#     # Save the resulting image
#     output_path = "Aligned/Aligned_ours_final_harry_potter2.png"
#     cv2.imwrite(output_path, stacked_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Save with no compression for maximum quality
#     print(f"Stacked image saved as {output_path}")
# else:
#     print("No images were loaded. Please check the image paths and indices.")
    
    
    
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
save_path="Aligned/Aligne_CelebA_comp7_supp_10_step.png"
# select_images=[3,7,25,31,32,64,69,81,82,88,102]
# select_images=[7,25,32,64,81,82,88,102]
# select_images=[7,25,541,64,81,82,556,570]
select_images=[25,541,64,81,570]   # crop half 
select_images=[25,541,64,81,570,50,104,166]    # checking more pose variations
select_images=[6,53,57,60,78,174,141]
select_images=[25,541,570,141,104]
# select_images=[129 , 64 ,106 ,115, 174 ,200 ,201 ,81 ,290,50 ,6] # This is for supplementary
select_images=[407,68,181,209]

# select_images=[7,24,32,64,69,81]
# select_images=[134,146,161,165,166,172]
# select_images=[3,7,25,31,32]
# select_images= [64,69,81,82,88,102]

path_list=[src_path,target_path,DiffFace_10_path,Diffswap_10_path,ours_5_path,ours_20_path]

Labels=["Source","Target","DiffFace-20","DiffSwap-20","Ours-5","Ours-20"]

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
    



images = read_images(path_list, select_images)
visualize_images(images,save_path)



Simswap_path="/home/sanoojan/other_swappers/SimSwap/output/FFHQ/results"
e4s_path="/home/sanoojan/e4s/Results/testbench/results_on_FFHQ_orig_ckpt/results"
DiffFace_path="/home/sanoojan/other_swappers/DiffFace/results/FFHQ/results"
Diffswap_path="/home/sanoojan/other_swappers/DiffSwap/all_images_with_folders_named_2_FFHQ"
HifiFace_path="/home/sanoojan/other_swappers/hififace/FFHQ_results"
FaceDancer_path="/home/sanoojan/other_swappers/FaceDancer/FaceDancer_c_HQ-FFHQ/results"
# ours_path="results_grad/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16_with_src_hair/results"
ours_path="results_FFHQ_FINAL/v5_Two_CLIP_proj_with_multiple_ID_losses_ep_17_3_75/results"
save_path="Aligned/Aligne_FFHQ_comp7_supp_extre.png"

src_path="dataset/FaceData/FFHQ/Val"
target_path="dataset/FaceData/FFHQ/Val_target"

# select_images=[3,7,25,31,32,64,69,81,82,88,102]
# select_images=[3,5,9,11,18,117,155,159]
select_images=[9,11,18,155,159]
select_images=[9,11,18,155,159,10,23,76,95,177,193]
select_images=[9,11,18,155,308]
select_images=[  4 ,  7, 101 ,193 ,203 ,213 ,177, 279 ,117 ,331 ,359] #supp
select_images=[686,792,19,960,300,303,310,329,374,375] # extreme cond

# select_images=[134,146,161,165,166,172]
# select_images=[3,7,25,31,32]
# select_images= [64,69,81,82,88,102]


path_list=[src_path,target_path,Simswap_path,e4s_path,FaceDancer_path,DiffFace_path,Diffswap_path,ours_path]

Labels=["Source","Target","SimSwap","E4s","FaceDancer","DiffFace","DiffSwap","Ours"]

# images = read_images(path_list, select_images)
# visualize_images(images,save_path)



############################Head Swap #########################

# Hair Swap Results 

# Ours_no_grad_path="results/v4_reconstruct_img_train_2_step_ep_38/results"
# Ours_grad_path="results_grad/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16_with_src_hair/results"

# # ours_path="results_grad/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16_with_src_hair/results"
# ours_path="results_grad/50_v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_noclip_same_image_ep20/results"
Ours_old_path="results_FINAL/v5_Two_CLIP_proj_with_multiple_ID_losses_ep12_75_hair_swap/results"
Ours_path="results_FINAL/v5_SRC_CLIP_proj_with_multiple_ID_losses_random_masks_hair_swap_3_75/results"
Ours_final_path="results_FINAL/v5_Two_CLIP_proj_with_multiple_ID_losses_random_mask_ep_16_swap_hair/results"
src_path="dataset/FaceData/CelebAMask-HQ/Val"
target_path="dataset/FaceData/CelebAMask-HQ/Val_target"
Mask_path="results_FINAL/v5_SRC_CLIP_proj_with_multiple_ID_losses_random_masks_hair_swap_3_75/samples"


select_images=[61,81,82,83,111,163,225]



path_list=[src_path,target_path,Mask_path,Ours_old_path,Ours_final_path]

Labels=["Source","Target","Inpaint","Ours w/o\nmask shuffling","Ours"]

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
        # filter paths 
        if "samples" in path:
            filenames = [filename for filename in filenames if "inpaint" in filename]
        
        filenames = natsort.natsorted(filenames)  # Sort filenames naturally
        
        
        
        selected_filenames = [filenames[i] for i in select_images]
        images.append([cv2.cvtColor(cv2.imread(os.path.join(path, filename)), cv2.COLOR_BGR2RGB) for filename in selected_filenames])
    return images

def visualize_images(images):
    num_paths = len(images)
    num_images_per_path = len(images[0])
    x_size=num_paths*6
    y_size=num_images_per_path*6
    fig, ax = plt.subplots(num_images_per_path, num_paths,figsize=(x_size, y_size))

    for i in range(num_images_per_path):
        for j in range(num_paths):
            ax[i, j].imshow(images[j][i])
            ax[i, j].axis('off')
            if i == 0:
                ax[i, j].set_title(Labels[j], fontsize=40)
                
    plt.subplots_adjust(wspace=0, hspace=0)
    # save plot
    plt.savefig("Image_outputs/visualize_hair_swap.png")
    
    # plt.show()

# images = read_images(path_list, select_images)
# # save 


# visualize_images(images)


################################# compare clip dissentangle #####################


# Simswap_path="/home/sanoojan/other_swappers/SimSwap/output/CelebA/results"
# e4s_path="/home/sanoojan/e4s/Results/testbench/results_Original_ckpt_without_crop/results"
# DiffFace_path="/home/sanoojan/other_swappers/DiffFace/results/CelebA/results"
# # ours_path="results_grad/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16_with_src_hair/results"
ours_path="results_FFHQ_FINAL/v5_Two_CLIP_proj_with_multiple_ID_losses_random_mask_hair_FINAL_ep14_HairSwap_sc_3_75/results"

source_path="dataset/FaceData/CelebAMask-HQ/Val"
target_path="dataset/FaceData/CelebAMask-HQ/Val_target"
Nose_only_path="results_FINAL/v5_Two_CLIP_proj_with_multiple_ID_losses_ep13_75_nose_only/results"
Nose_and_mouth="results_FINAL/v5_Two_CLIP_proj_with_multiple_ID_losses_ep13_75_nose_and_mouth_only/results"
Nose_and_mouth_and_eyes="results_FINAL/v5_Two_CLIP_proj_with_multiple_ID_losses_ep13_75_eyes_nose_and_mouth_only/results"
Full_face="results_FINAL/v5_Two_CLIP_proj_with_multiple_ID_losses_ep13_75/results"
# select_images=[3,7,25,31,32,64,69,81,82,88,102]
# select_images=[14,24]
# select_images=[134,146,161,165,166,172]
# select_images=[3,7,25,31,32]
# select_images= [64,69,81,82,88,102]
select_images=[13,30,50,62,36,63,88]

path_list=[source_path,target_path,Nose_only_path,Nose_and_mouth,Nose_and_mouth_and_eyes,Full_face]

Labels=["Source","Target","Nose_only","Nose_and_mouth","Nose_and_mouth_and_eyes","Full_face"]

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import natsort

def read_images(path_list, select_images):
    images = []
    for path in path_list:
        filenames = os.listdir(path)
        filenames = natsort.natsorted(filenames)  # Sort filenames naturally
        selected_filenames = [filenames[i] for i in select_images]
        images.append([cv2.cvtColor(cv2.imread(os.path.join(path, filename)), cv2.COLOR_BGR2RGB) for filename in selected_filenames])
    return images

def visualize_images(images):
    num_paths = len(images)
    num_images_per_path = len(images[0])
    x_size=num_paths*5
    y_size=num_images_per_path*5
    fig, ax = plt.subplots(num_images_per_path, num_paths,figsize=(x_size, y_size))

    for i in range(num_images_per_path):
        for j in range(num_paths):
            ax[i, j].imshow(images[j][i])
            ax[i, j].axis('off')
            if i == 0:
                ax[i, j].set_title(Labels[j], fontsize=20)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    # save plot
    plt.savefig("Image_outputs/clip_dissentanglement.png", bbox_inches='tight', pad_inches=0)
    # plt.show()

# images = read_images(path_list, select_images)
# visualize_images(images)