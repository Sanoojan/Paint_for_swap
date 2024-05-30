
# Set variables
name="24_v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_src_features_ep7"
Results_dir="results_grad/${name}"
Results_out="results_grad/${name}/results"
Write_results="Quantitative_grad/P4s/${name}"
device=4

CONFIG="models/Paint-by-Example/Final_src_clip_ID_features_Text_eq_emb_stable_diff/PBE/celebA/2024-02-24T17-26-24_v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_src_features/configs/2024-02-24T17-26-24-project.yaml"
CKPT="models/Paint-by-Example/Final_src_clip_ID_features_Text_eq_emb_stable_diff/PBE/celebA/2024-02-24T17-26-24_v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_src_features/checkpoints/last.ckpt"
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

CUDA_VISIBLE_DEVICES=${device} python scripts/inference_test_bench.py \
    --outdir "${Results_dir}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --scale 5 \
    --n_samples 10 \
    --device_ID ${device} \
    --dataset "CelebA" \
    --ddim_steps 50


    # --Start_from_target \
    # --target_start_noise_t 1000  
    


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


