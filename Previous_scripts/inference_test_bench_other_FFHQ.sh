
# Set variables
Results_out="/home/sanoojan/other_swappers/SimSwap/output/FFHQ/results"
# Results_out="results_FFHQ/v4_reconstruct_img_train_with_DIFT_prior_1000/results"
Write_results="results_FFHQ/quantitative/SimSwap/512_model"

# Set variables
# name="avg_3_features_full_face_with_augs_scale1"

# Results_out="/home/sanoojan/e4s/Results/testbench/results_CelebA_ckpt_without_crop"
# Write_results="results/quantitative/P4s/${name}"
# Write_results="results/quantitative/P4s/check-target_id"
device=1


# CONFIG="configs/v3_Landmark_cond.yaml"
# CKPT="models/Paint-by-Example/Arcface_features_clip_avg_landmarks_full_face_mask_new_augs/PBE/celebA/2023-10-04T10-53-42_v3_Landmark_cond/checkpoints/last.ckpt"
source_path="dataset/FaceData/FFHQ/Val"
target_path="dataset/FaceData/FFHQ/Val_target"
source_mask_path="dataset/FaceData/FFHQ/src_mask"
target_mask_path="dataset/FaceData/FFHQ/target_mask"
Dataset_path="dataset/FaceData/FFHQ/images512"

current_time=$(date +"%Y%m%d_%H%M%S")
output_filename="${Write_results}/out_${current_time}.txt"

if [ ! -d "$Write_results" ]; then
    mkdir -p "$Write_results"
    echo "Directory created: $Write_results"
else
    echo "Directory already exists: $Write_results"
fi


# Run FID score calculation

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
    "${Results_out}"  >> "$output_filename"

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

