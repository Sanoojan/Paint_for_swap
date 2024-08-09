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
Results_out="/home/sanoojan/other_swappers/DiffSwap/all_images_with_folders_named_2_celeba_20"
# Results_out="/home/sanoojan/other_swappers/DiffFace/results/Celeba_20/results"
res_end="results"
results_start="results_grad"
Write_results="Quantitative_Grad_all"
device=0

# declare -a names=("avg_3_features_full_face_with_augs_scale1"
#                     "check"
#                     "clip_ID_landmark_avg_with_exact_mask_trained_inf" 
#                     "clip_ID_landmark_avg_with_neck"
#                      "clip_ID_landmark_avg_with_neck_scale10" 
#                      "Done_quantitative_eval/avg_3_features_full_face_with_augs" 
#                      "Done_quantitative_eval/clip_ID_landmarks_avg"
#                      "Done_quantitative_eval/Dont_know"
#                     "Done_quantitative_eval/Finetuned"
#                     "Done_quantitative_eval/Paint_by_example_original"
#                      "v3_Landmark_cond_with_DIFT_recon_1000_noise_check"
#                      "id_clip_landmark_avg_with_10"
#                      "ID_Landmark_CLIP_reconstruct_img_train_noise_750"
#                     "ID_Landmark_CLIP_reconstruct_img_train_noise_1000"
#                     "original_DIFT_recon_700"
#                     "v2_clip_cond_hair_swap"
#                     "v3_Landmark_cond_with_DIFT_recon_1000_noise_check"
#                     "v3_reconstruct_img_train_hair_swap"
#                     "v4_img_train_2_step_multi_false"
#                     "v4_reconstruct_img_train_1_step"
#                     "v4_reconstruct_img_train_1_step_cross_attn_only"
#                     "v4_reconstruct_img_train_1_step_cross_attn_only_34"
#                     "v4_reconstruct_img_train_1_step_cross_attn_only_40"
#                     "v4_reconstruct_img_train_1_step_ep_9"
#                     "v4_reconstruct_img_train_1_step_ep_14"
#                     "v4_reconstruct_img_train_1_step_ep_24"
#                     "v4_reconstruct_img_train_2_step_ep_29"
#                     "v4_reconstruct_img_train_2_step_ep_38"
#                     "v4_reconstruct_img_train_2_step_multi_false"
#                     "v4_reconstruct_img_train_2_step_multi_false_50"
#                     "v4_reconstruct_img_train_2_step_multi_false_onlyID_36"
#                     "v4_reconstruct_img_train_4_step_multi_false_13"
#                     "v4_reconstruct_img_train_4_step_multi_false_with_LPIPS_33"
#                     "v4_reconstruct_img_train_4_step_multi_false_with_LPIPS_47"
#                     "v4_reconstruct_img_train_correct"
#                     "v4_reconstruct_img_train_correct_id"
#                     "v4_reconstruct_img_train_correct_ID_ch"
#                     "v4_reconstruct_img_train_correct_id_from_reenact"
#                     "v4_reconstruct_img_train_correct_no_lpips_0.9ID"
#                     "v4_reconstruct_img_train_correct_normalize"
#                     "v4_reconstruct_img_train_correct17"
#                     "v4_reconstruct_img_train_reducing_clip_20_ep"
#                     "v4_reconstruct_img_train_reducing_clip_20_ep_1000_tar"
#                     "v4_reconstruct_img_train_with_DIFT_recon_600_noise"
#                     "v4_reconstruct_img_train_with_DIFT_recon_750_noise"
#                     "v4_reconstruct_img_train_with_DIFT_recon_1000_noise"
#                     "v4_reconstruct_img_train_with_neck"
#                     "v4_reconstruct_img_train_with_neck_DDIM"
#                     "v4_reconstruct_img_train_with_neck_DDIM_1000_uc0"
#                     "v4_reconstruct_img_train_with_neck_DIFT_700"
#                     "v4_reconstruct_img_train_with_neck_DIFT_1000"
#                     "v8_concat_feat"
#                     "v9_add_feat"
#                     "v9_add_feat_normalize_False_multi_scale_False"
#                     "v9_add_feat_normalize_False_multi_scale_False_0.9_ID"
#                     "v9_add_feat_normalize_True_multi_scale_False"
#                     "12_reconstruct_img_train_correct_sep_head_att"
#                      )

# declare -a names=("50_v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_noclip_same_image_ep06"
#                         "50_v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_noclip_same_image_ep09"
#                         "50_v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_noclip_same_image_ep12"
#                         "50_v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_noclip_same_image_ep20"
#                         "v4_2_step_LPIPS_stack_ep10"
#                         "v4_2_step_LPIPS_stack_ep2"
#                         "v4_2_step_LPIPS_stack_ep3"
#                         "v4_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT_LPIPS_ep_17"
#                         "v4_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT_LPIPS_ep_19"
#                         "v4_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT_LPIPS_ep_20"
#                         "v4_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT_LPIPS_ep_21"
#                         "v4_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT_LPIPS_ep_23"
#                         "v4_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT_LPIPS_ep_24"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_DDIM"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_DDIM_2step"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep10"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep11"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep12"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep13"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16_clip_disentanglement_test"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16_target_clip_feat"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16_target_clip_feat_dissentangle_test_1"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16_target_eye_mth_scale1"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16_target_eye_mth_scale5"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep16_with_src_hair"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep3"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep5"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep6"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep7"
#                         "v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_ep9"
#                     )



# Set variables
# name="avg_3_features_full_face_with_augs_scale1"

# Results_out="/home/sanoojan/e4s/Results/testbench/results_CelebA_ckpt_without_crop"
# Write_results="results/quantitative/P4s/${name}"
# Write_results="results/quantitative/P4s/check-target_id"
device=0


# CONFIG="configs/v3_Landmark_cond.yaml"
# CKPT="models/Paint-by-Example/Arcface_features_clip_avg_landmarks_full_face_mask_new_augs/PBE/celebA/2023-10-04T10-53-42_v3_Landmark_cond/checkpoints/last.ckpt"
source_path="dataset/FaceData/CelebAMask-HQ/Val"
target_path="dataset/FaceData/CelebAMask-HQ/Val_target"
source_mask_path="dataset/FaceData/CelebAMask-HQ/src_mask"
target_mask_path="dataset/FaceData/CelebAMask-HQ/target_mask"
Dataset_path="dataset/FaceData/CelebAMask-HQ/CelebA-HQ-img"

# for name in "${names[@]}"
# do
#     Results_out="${results_start}/${name}/${res_end}"
#     current_time=$(date +"%Y%m%d_%H%M%S")
#     Write_results_n="${Write_results}/${name}"
#     output_filename="${Write_results_n}/out_${current_time}.txt"
    
#     if [ ! -d "$Write_results_n" ]; then
#         mkdir -p "$Write_results_n"
#         echo "Directory created: $Write_results_n"
#     else
#         echo "Directory already exists: $Write_results_n"
#     fi

#     # Run FID score calculation
#     # ... (rest of your script)
#     CUDA_VISIBLE_DEVICES=${device} python eval_tool/fid/fid_score.py --device cuda \
#         "${source_path}" \
#         "${Results_out}"  >> "$output_filename"

#     # echo "FID score with Dataset:"
#     CUDA_VISIBLE_DEVICES=${device} python eval_tool/fid/fid_score.py --device cuda \
#         "${Dataset_path}" \
#         "${Results_out}"  >> "$output_filename"

#     # echo "Pose comarison with target:"
#     CUDA_VISIBLE_DEVICES=${device} python eval_tool/Pose/pose_compare.py --device cuda \
#         "${target_path}" \
#         "${Results_out}"  >> "$output_filename"


#     # echo "ID similarity with Target:"
#     CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
#         "${target_path}" \
#         "${Results_out}" \
#         "${target_mask_path}" \
#         "${target_mask_path}" >> "$output_filename"  

#     # echo "ID_restoreformer"
#     CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_distance.py  \
#         "${Results_out}" \
#         --gt_folder "${source_path}" >> "$output_filename"  

#     echo "Expression comarison with target:"
#     CUDA_VISIBLE_DEVICES=${device} python eval_tool/Expression/expression_compare_face_recon.py --device cuda \
#         "${target_path}" \
#         "${Results_out}"  >> "$output_filename"


#     echo "ID similarity with Source:"
#     CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
#         "${source_path}" \
#         "${Results_out}" \
#         "${source_mask_path}" \
#         "${target_mask_path}"  \
#         --print_sim True  >> "$output_filename"  
# done


    output_filename="${Write_results}/out_${current_time}.txt"

    echo "FID score with Source:"   >> "$output_filename"       
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/fid/fid_score.py --device cuda \
        "${source_path}" \
        "${Results_out}"  >> "$output_filename"

    echo "FID score with Dataset:" >> "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/fid/fid_score.py --device cuda \
        "${Dataset_path}" \
        "${Results_out}"  >> "$output_filename"

    echo "Pose comarison with target:" >> "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/Pose/pose_compare.py --device cuda \
        "${target_path}" \
        "${Results_out}"  >> "$output_filename"

    echo "Expression comarison with target:" >> "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/Expression/expression_compare_face_recon.py --device cuda \
        "${target_path}" \
        "${Results_out}" >> "$output_filename"

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
        --print_sim True  >> "$output_filename"   


    echo "ID similarity with Source using Arcface:" >> "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
        "${source_path}" \
        "${Results_out}" \
        "${source_mask_path}" \
        "${target_mask_path}" \
        --print_sim True  \
        --arcface True >> "$output_filename" 

 