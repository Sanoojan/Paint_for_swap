
# Set variables
name="check"
Results_dir="results/${name}"
Results_out="results/${name}/results"
Write_results="results/quantitative/P4s/${name}"
device=1

CONFIG="configs/v7_reconstruct_img_train_concat_feat.yaml"
CKPT="models/Paint-by-Example/v7_reconstruct_img_train_concat_feat/PBE/celebA/2023-10-27T00-54-41_v7_reconstruct_img_train_concat_feat/checkpoints/last.ckpt"
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
    --plms \
    --outdir "${Results_dir}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --scale 5 \
    --dataset "CelebA" 


    # --Start_from_target \
    # --target_start_noise_t 700  
    


echo "FID score with Source:"
# CUDA_VISIBLE_DEVICES=${device} python eval_tool/fid/fid_score.py --device cuda \
#     "${source_path}" \
#     "${Results_out}"  >> "$output_filename"

# echo "FID score with Dataset:"
# CUDA_VISIBLE_DEVICES=${device} python eval_tool/fid/fid_score.py --device cuda \
#     "${Dataset_path}" \
#     "${Results_out}"  >> "$output_filename"

# echo "Pose comarison with target:"
# CUDA_VISIBLE_DEVICES=${device} python eval_tool/Pose/pose_compare.py --device cuda \
#     "${target_path}" \
#     "${Results_out}"  >> "$output_filename"

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

echo "ID_restoreformer"
CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_distance.py  \
    "/home/sanoojan/e4s/Results/testbench/results_Original_ckpt_without_crop/results" \
    --gt_folder "${source_path}"   >> "$output_filename"  