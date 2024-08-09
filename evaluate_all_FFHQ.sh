#!/bin/bash

# Set variables
# Results_out="results/v4_reconstruct_img_train_with_neck_DDIM/results"
# Results_out="results/v4_reconstruct_img_train_correct_ID_ch/results"
# Results_out="/home/sanoojan/other_swappers/SimSwap/output/CelebA/results" # 51.6, 72.8
# Results_out="/home/sanoojan/e4s/Results/testbench/results_Original_ckpt_without_crop/results" #38.3, 57.3
# Results_out="/home/sanoojan/other_swappers/DiffFace/results/CelebA/results" # 38.3, 57.3
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
res_end="results"
results_start=""
Write_results="Quantitative_Grad_Other_swappers/FFHQ"



declare -a names=(  
                    # "/home/sanoojan/other_swappers/MegaFs/FFHQ_outs"
                    # "/home/sanoojan/other_swappers/hififace/FFHQ_results"
                    "/home/sanoojan/e4s/Results/testbench/results_on_FFHQ_orig_ckpt/results"                
                    # "/home/sanoojan/other_swappers/SimSwap/output/FFHQ/results"
                    # "/home/sanoojan/other_swappers/FaceDancer/FaceDancer_c_HQ-FFHQ/results"
                    "/home/sanoojan/other_swappers/DiffSwap/all_images_with_folders_named_2_FFHQ"
                    # "/home/sanoojan/other_swappers/DiffFace/results/FFHQ/results"
                    )


# Set variables
# name="avg_3_features_full_face_with_augs_scale1"

# Results_out="/home/sanoojan/e4s/Results/testbench/results_CelebA_ckpt_without_crop"
# Write_results="results/quantitative/P4s/${name}"
# Write_results="results/quantitative/P4s/check-target_id"
device=0


# CONFIG="configs/v3_Landmark_cond.yaml"
# CKPT="models/Paint-by-Example/Arcface_features_clip_avg_landmarks_full_face_mask_new_augs/PBE/celebA/2023-10-04T10-53-42_v3_Landmark_cond/checkpoints/last.ckpt"
source_path="dataset/FaceData/FFHQ/Val"
target_path="dataset/FaceData/FFHQ/Val_target"
source_mask_path="dataset/FaceData/FFHQ/src_mask"
target_mask_path="dataset/FaceData/FFHQ/target_mask"
Dataset_path="dataset/FaceData/FFHQ/images512"

for name in "${names[@]}"
do
    # Results_out="${results_start}/${name}/${res_end}"
    # Results_out="${name}/${res_end}"
    Results_out="${name}"
    current_time=$(date +"%Y%m%d_%H%M%S")
    Write_results_n="${Write_results}/${name}"
    output_filename="${Write_results_n}/out_${current_time}.txt"
    
    if [ ! -d "$Write_results_n" ]; then
        mkdir -p "$Write_results_n"
        echo "Directory created: $Write_results_n"
    else
        echo "Directory already exists: $Write_results_n"
    fi

    # echo "FID score with Source:"   >> "$output_filename"
    # CUDA_VISIBLE_DEVICES=${device} python eval_tool/fid/fid_score.py --device cuda \
    #     "${source_path}" \
    #     "${Results_out}"  >> "$output_filename"

    # echo "FID score with Dataset:" >> "$output_filename"
    # CUDA_VISIBLE_DEVICES=${device} python eval_tool/fid/fid_score.py --device cuda \
    #     "${Dataset_path}" \
    #     "${Results_out}"  >> "$output_filename"

    # echo "Pose comarison with target:" >> "$output_filename"
    # CUDA_VISIBLE_DEVICES=${device} python eval_tool/Pose/pose_compare.py --device cuda \
    #     "${target_path}" \
    #     "${Results_out}"  >> "$output_filename"

    echo "Expression comarison with target:" >> "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/Expression/expression_compare_face_recon.py --device cuda \
        "${target_path}" \
        "${Results_out}" 
        # >> "$output_filename"

    echo "ID similarity with Target:" >> "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
        "${target_path}" \
        "${Results_out}" \
        "${target_mask_path}" \
        "${target_mask_path}"  >> "$output_filename"  

    echo "ID_restoreformer" >> "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_distance.py  \
        "${Results_out}" \
        --gt_folder "${source_path}"   >> "$output_filename"  

    echo "ID similarity with Source using cosface:" >> "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
        "${source_path}" \
        "${Results_out}" \
        "${source_mask_path}" \
        "${target_mask_path}" \
        --dataset "ffhq" \
        --print_sim True  >> "$output_filename"   


    echo "ID similarity with Source using Arcface:" >> "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
        "${source_path}" \
        "${Results_out}" \
        "${source_mask_path}" \
        "${target_mask_path}" \
        --dataset "ffhq" \
        --print_sim True  \
        --arcface True >> "$output_filename" 
done
