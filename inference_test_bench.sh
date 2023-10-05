
# Set variables
name="clip_ID_landmark_avg_with_exact_mask_trained_inf"
Results_out="results/${name}"
# Write_results="results/quantitative/P4s/${name}"
Write_results="results/quantitative/P4s/check-target_id"
device=1

CONFIG="configs/v3_Landmark_cond.yaml"
CKPT="models/Paint-by-Example/Arcface_features_clip_avg_landmarks/PBE/celebA/2023-09-28T02-45-37_v3_Landmark_cond/checkpoints/last.ckpt"
source_path="dataset/FaceData/CelebAMask-HQ/Val"
source_path="dataset/FaceData/CelebAMask-HQ/Val_target"


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
#     --scale 5

# Run FID score calculation
# CUDA_VISIBLE_DEVICES=${device} python eval_tool/fid/fid_score.py --device cuda \
#     "${source_path}" \
#     "${Results_out}/results"  >> "$output_filename"

# CUDA_VISIBLE_DEVICES=${device} python eval_tool/Pose/pose_compare.py --device cuda \
#     "${target_path}" \
#     "${Results_out}/results"  >> "$output_filename"
     
CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
    "${source_path}" \
    "${Results_out}/results"   >> "$output_filename"  

