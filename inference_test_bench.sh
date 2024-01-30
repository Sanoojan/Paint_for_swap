
# Set variables
name="v4_reconstruct_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT_39"
Results_dir="results_new/${name}"
Results_out="results_new/${name}/results"
Write_results="Quantitative_new/P4s/${name}"
device=1

CONFIG="models/Paint-by-Example/v4_reconstruct_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT/PBE/celebA/2024-01-24T13-41-55_v4_reconstruct_img_train_2_step_multi_false/configs/2024-01-24T13-41-55-project.yaml"
CKPT="models/Paint-by-Example/v4_reconstruct_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT/PBE/celebA/2024-01-24T13-41-55_v4_reconstruct_img_train_2_step_multi_false/checkpoints/epoch=000039.ckpt"
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

# Run inference

python scripts/inference_test_bench.py \
    --outdir "${Results_dir}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --scale 5 \
    --n_samples 10 \
    --device_ID ${device} \
    --dataset "CelebA" 

    # --Start_from_target \
    # --target_start_noise_t 800  
    


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


echo "ID similarity with Target:"
CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
    "${target_path}" \
    "${Results_out}" \
    "${target_mask_path}" \
    "${target_mask_path}"  >> "$output_filename"  

echo "ID_restoreformer"
CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_distance.py  \
    "${Results_out}" \
    --gt_folder "${source_path}"   >> "$output_filename"  

echo "ID similarity with Source:"
CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
    "${source_path}" \
    "${Results_out}" \
    "${source_mask_path}" \
    "${target_mask_path}" \
    --print_sim True  >> "$output_filename"   

