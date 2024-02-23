
# Set variables
name="v4_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT_LPIPS_ep0_fazli_to_jc"
Results_dir="results_video/${name}"
Base_dir="results_video"
Results_out="results_video/${name}/results"
# Write_results="results/quantitative/P4s/${name}"
device=5

CONFIG="models/Paint-by-Example/v4_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT_LPIPS/PBE/celebA/2024-02-05T22-04-36_v4_reconstruct_img_train_2_step_multi_false_with_LPIPS/configs/2024-02-05T22-04-36-project.yaml"
CKPT="models/Paint-by-Example/v4_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT_LPIPS/PBE/celebA/2024-02-05T22-04-36_v4_reconstruct_img_train_2_step_multi_false_with_LPIPS/checkpoints/epoch=000017.ckpt"

source_path="dataset/FaceData/CelebAMask-HQ/Val"
source_mask_path="dataset/FaceData/CelebAMask-HQ/src_mask"


current_time=$(date +"%Y%m%d_%H%M%S")
output_filename="${Write_results}/out_${current_time}.txt"

# if [ ! -d "$Write_results" ]; then
#     mkdir -p "$Write_results"
#     echo "Directory created: $Write_results"
# else
#     echo "Directory already exists: $Write_results"
# fi

# Run inference

CUDA_VISIBLE_DEVICES=${device} python scripts/inference_swap_video.py \
    --outdir "${Results_dir}" \
    --target_video "examples/faceswap/JhonCena.mp4" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --src_image "examples/faceswap/IMG_4873.png" \
    --Base_dir "${Base_dir}" \
    --scale 5 


    # --Start_from_target \
    # --target_start_noise_t 1000  
    

