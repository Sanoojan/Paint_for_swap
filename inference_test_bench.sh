
# Set variables
name="avg_3_features_full_face_with_augs_scale1"
Results_out="results/${name}"
# Results_out="/home/sanoojan/e4s/Results/testbench/results_CelebA_ckpt_without_crop"
Write_results="results/quantitative/P4s/${name}"
# Write_results="results/quantitative/P4s/check-target_id"
device=2

CONFIG="configs/v3_Landmark_cond.yaml"
CKPT="models/Paint-by-Example/Arcface_features_clip_avg_landmarks_full_face_mask_new_augs/PBE/celebA/2023-10-04T10-53-42_v3_Landmark_cond/checkpoints/last.ckpt"
source_path="dataset/FaceData/CelebAMask-HQ/Val"
target_path="dataset/FaceData/CelebAMask-HQ/Val_target"
source_mask_path="dataset/FaceData/CelebAMask-HQ/src_mask"
target_mask_path="dataset/FaceData/CelebAMask-HQ/target_mask"


current_time=$(date +"%Y%m%d_%H%M%S")
output_filename="${Write_results}/out_${current_time}.txt"

if [ ! -d "$Write_results" ]; then
    mkdir -p "$Write_results"
    echo "Directory created: $Write_results"
else
    echo "Directory already exists: $Write_results"
fi

# Run inference
# CUDA_VISIBLE_DEVICES=${device} python scripts/inference_test_bench.py \
#     --plms \
#     --outdir "${Results_out}" \
#     --config "${CONFIG}" \
#     --ckpt "${CKPT}" \
#     --scale 1

# Run FID score calculation
CUDA_VISIBLE_DEVICES=${device} python eval_tool/fid/fid_score.py --device cuda \
    "${source_path}" \
    "${Results_out}/results"  >> "$output_filename"

CUDA_VISIBLE_DEVICES=${device} python eval_tool/Pose/pose_compare.py --device cuda \
    "${target_path}" \
    "${Results_out}/results"  >> "$output_filename"
     
CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
    "${source_path}" \
    "${Results_out}/results" \
    "${source_mask_path}" \
    "${target_mask_path}"  >> "$output_filename"  

# CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
#     "${target_path}" \
#     "${Results_out}/results" \
#     "${target_mask_path}" \
#     "${source_mask_path}"  >> "$output_filename"  
