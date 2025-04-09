
# Set variables
name="v5_Two_CLIP_proj_with_multiple_ID_losses_ep16_3_75"
Results_dir="results_FF++/${name}"
Results_out="results_FF++/${name}/results"
Write_results="results_FF++/quantitative/P4s/${name}"
device=3

CONFIG="models/Paint-by-Example/v5_Two_CLIP_proj_with_multiple_ID_losses/PBE/celebA/2024-02-27T00-54-33_v5_Two_CLIP_proj_with_multiple_ID_losses/configs/2024-02-27T00-54-33-project.yaml"
CKPT="models/Paint-by-Example/v5_Two_CLIP_proj_with_multiple_ID_losses/PBE/celebA/2024-02-27T00-54-33_v5_Two_CLIP_proj_with_multiple_ID_losses/checkpoints/last.ckpt"
source_path="dataset/FaceData/FF++/Val"
target_path="dataset/FaceData/FF++/Val_target"
source_mask_path="dataset/FaceData/FF++/src_mask"
target_mask_path="dataset/FaceData/FF++/target_mask"
Dataset_path="dataset/FaceData/FF++/faceforensics_benchmark_images"
Dataset_dir="dataset/FaceData/FF++"

current_time=$(date +"%Y%m%d_%H%M%S")
output_filename="${Write_results}/out_${current_time}.txt"

if [ ! -d "$Write_results" ]; then
    mkdir -p "$Write_results"
    echo "Directory created: $Write_results"
else
    echo "Directory already exists: $Write_results"
fi

# Run inference

CUDA_VISIBLE_DEVICES=${device} python scripts/inference_test_bench.py \
    --outdir "${Results_dir}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --scale 3 \
    --n_samples 10 \
    --device_ID ${device} \
    --dataset "FF++" \
    --dataset_dir "${Dataset_dir}" \
    --ddim_steps 75





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
    --print_sim True  >> "$output_filename"   


echo "ID similarity with Source using Arcface:" >> "$output_filename"
CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
    "${source_path}" \
    "${Results_out}" \
    "${source_mask_path}" \
    "${target_mask_path}" \
    --print_sim True  \
    --arcface True >> "$output_filename"   


