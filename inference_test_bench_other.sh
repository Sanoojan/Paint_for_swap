
# Set variables
# Results_out="results/v4_reconstruct_img_train_with_neck_DDIM/results"
# Results_out="results/v4_reconstruct_img_train_correct_ID_ch/results"
# Results_out="/home/sanoojan/other_swappers/SimSwap/output/CelebA/results" # 51.6, 72.8
Results_out="/home/sanoojan/e4s/Results/testbench/results_Original_ckpt_without_crop/results" #38.3, 57.3
# Results_out="" # 38.3, 57.3
# Results_out="results/v9_add_feat/results"
# Results_out="results/v4_reconstruct_img_train_with_DIFT_recon_750_noise/results"
# Results_out="results/v4_reconstruct_img_train_with_DIFT_recon_1000_noise/results"
# Results_out="results/v4_reconstruct_img_train_with_DDIM/results"
# Results_out="/home/sanoojan/e4s/Results/testbench/reenact/results"
# Results_out="intermediate_renact/results" 
# Results_out="results/v9_add_feat/results"
# Results_out="results/v4_reconstruct_img_train_correct_id_from_reenact/results"
# Results_out="results/v4_reconstruct_img_train_correct_id/results"       # 11.7,24,8,0.31
# Results_out="results/v4_reconstruct_img_train_correct_id_from_reenact/results"    # 15.4,32.1,0.33
# Results_out="results/v4_reconstruct_img_train_2_step_ep_38/results"
# Results_out="/home/sanoojan/other_swappers/FaceDancer/FaceDancer_c_HQ/results"
# Results_out="results_grad/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS/results"
# Results_out="results_grad/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_DDIM/results"
# Results_out="results_new/v4_reconstruct_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT_49/results"
# Results_out="results_grad/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep10/results"
Results_out="/home/sanoojan/other_swappers/DiffFace/results/CelebA/results"
Results_out="results_grad/Target_CLIP_SRC_ID_ep11_no_hair/results"


Write_results="results/Debug/with_grad_trained"

# Set variables
# name="avg_3_features_full_face_with_augs_scale1"

# Results_out="/home/sanoojan/e4s/Results/testbench/results_CelebA_ckpt_without_crop"
# Write_results="results/quantitative/P4s/${name}"
# Write_results="results/quantitative/P4s/check-target_id"
device=2

CONFIG="configs/v3_Landmark_cond.yaml"
CKPT="models/Paint-by-Example/Arcface_features_clip_avg_landmarks_full_face_mask_new_augs/PBE/celebA/2023-10-04T10-53-42_v3_Landmark_cond/checkpoints/last.ckpt"
source_path="dataset/FaceData/CelebAMask-HQ/Val"
target_path="dataset/FaceData/CelebAMask-HQ/Val_target"
source_mask_path="dataset/FaceData/CelebAMask-HQ/src_mask"
target_mask_path="dataset/FaceData/CelebAMask-HQ/target_mask"
Dataset_path="dataset/FaceData/CelebAMask-HQ/CelebA-HQ-img"

current_time=$(date +"%Y%m%d_%H%M%S")
output_filename="${Write_results}/out_${current_time}.txt"

if [ ! -d "$Write_results" ]; then
    mkdir -p "$Write_results"
    echo "Directory created: $Write_results"
else
    echo "Directory already exists: $Write_results"
fi


# Run FID score calculation

# echo "FID score with Source:"
# CUDA_VISIBLE_DEVICES=${device} python eval_tool/fid/fid_score.py --device cuda \
#     "${source_path}" \
#     "${Results_out}"  >> "$output_filename"

# echo "FID score with Dataset:"
# CUDA_VISIBLE_DEVICES=${device} python eval_tool/fid/fid_score.py --device cuda \
#     "${Dataset_path}" \
#     "${Results_out}"  >> "$output_filename"

# echo "Pose comarison with target:"
# CUDA_VISIBLE_DEVICES=${device} python eval_tool/Pose/pose_compare.py --device cuda \
#     "${target_path}" \
#     "${Results_out}"  >> "$output_filename"

# # echo "Expression comarison with target:"
# CUDA_VISIBLE_DEVICES=${device} python eval_tool/Expression/expression_compare_face_recon.py --device cuda \
#     "${target_path}" \
#     "${Results_out}" \
#     --print_sim True  >> "$output_filename"


# echo "ID similarity with Target:"
# CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
#     "${target_path}" \
#     "${Results_out}" \
#     "${target_mask_path}" \
#     "${target_mask_path}" >> "$output_filename"  

# echo "ID_restoreformer"
# CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_distance.py  \
#     "${Results_out}" \
#     --gt_folder "${source_path}" >> "$output_filename"  

# echo "ID similarity with Source:"
CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
    "${source_path}" \
    "${Results_out}" \
    "${source_mask_path}" \
    "${target_mask_path}"  \
    --print_sim False  
    
    # >> "$output_filename"  