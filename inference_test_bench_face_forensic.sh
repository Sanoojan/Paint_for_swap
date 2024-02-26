
# Set variables
name="Tar_to_SRC_conf_FFHQ_trained"
Results_dir="results_FF++/${name}"
Results_out="results_FF++/${name}/results"
Write_results="results_FF++/quantitative/P4s/${name}"
device=2

CONFIG="models_from_CIAI/FFHQ/Tar_to_SRC_conf/configs/project_FF.yaml"
CKPT="models_from_CIAI/FFHQ/Tar_to_SRC_conf/checkpoints/last.ckpt"
source_path="dataset/FaceData/FF++/Val"
target_path="dataset/FaceData/FF++/Val_target"
source_mask_path="dataset/FaceData/FF++/src_mask"
target_mask_path="dataset/FaceData/FF++/target_mask"
Dataset_path="dataset/FaceData/FF++/faceforensics_benchmark_images"

current_time=$(date +"%Y%m%d_%H%M%S")
output_filename="${Write_results}/out_${current_time}.txt"

if [ ! -d "$Write_results" ]; then
    mkdir -p "$Write_results"
    echo "Directory created: $Write_results"
else
    echo "Directory already exists: $Write_results"
fi

# Run inference

python scripts/inference_test_bench.py \
    --outdir "${Results_dir}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --scale 5 \
    --n_samples 15 \
    --device_ID ${device} \
    --dataset "FF++" \
    --ddim_steps 50


    # --target_start_noise_t 700 \
    # --Start_from_target  


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