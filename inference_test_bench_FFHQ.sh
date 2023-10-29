
# Set variables
name="v4_reconstruct_img_train_with_DIFT_prior_700"
Results_dir="results_FFHQ/${name}"
Results_out="results_FFHQ/${name}/results"
Write_results="results_FFHQ/quantitative/P4s/${name}"
device=2

CONFIG="configs/v4_reconstruct_img_train_ffhq.yaml"
CKPT="models/Paint-by-Example/ID_Landmark_CLIP_reconstruct_img_train/PBE/celebA/2023-10-07T21-09-06_v4_reconstruct_img_train/checkpoints/last.ckpt"
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

# Run inference

CUDA_VISIBLE_DEVICES=${device} python scripts/inference_test_bench.py \
    --plms \
    --outdir "${Results_dir}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --scale 5 \
    --dataset "FFHQ" \
    --target_start_noise_t 700 \
    --Start_from_target  


echo "FID score with Source:"
CUDA_VISIBLE_DEVICES=${device} python eval_tool/fid/fid_score.py --device cuda \
    "${source_path}" \
    "${Results_out}"  >> "$output_filename"

echo "FID score with Dataset:"
CUDA_VISIBLE_DEVICES=${device} python eval_tool/fid/fid_score.py --device cuda \
    "${Dataset_path}" \
    "${Results_out}"  >> "$output_filename"

echo "Pose comarison with target:"
CUDA_VISIBLE_DEVICES=${device} python eval_tool/Pose/pose_compare.py --device cuda \
    "${target_path}" \
    "${Results_out}"  >> "$output_filename"

echo "ID similarity with Source:"
CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
    "${source_path}" \
    "${Results_out}" \
    "${source_mask_path}" \
    "${target_mask_path}"  >> "$output_filename"  


echo "ID similarity with Target:"
CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
    "${target_path}" \
    "${Results_out}" \
    "${target_mask_path}" \
    "${target_mask_path}"  >> "$output_filename"  