
# Set variables
name="v4_reconstruct_img_train_correct_id"
Results_dir="results/${name}"
Results_out="results/${name}/results"
Write_results="results/quantitative/P4s/${name}"
device=0

CONFIG="configs/v4_reconstruct_img_train_correct.yaml"
CKPT="models/Paint-by-Example/v4_reconstruct_img_train_correct_id/PBE/celebA/2023-11-05T18-28-26_v4_reconstruct_img_train_correct/checkpoints/last.ckpt"
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



echo "ID similarity with Source:"
CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
    "${Dataset_path}" \
    "Reverse_find_folder" \
    "dataset/FaceData/CelebAMask-HQ/CelebA-HQ-mask/Overall_mask" \
    "${target_mask_path}" 


