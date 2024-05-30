
# Set variables
name="v5_elon_to_news_ep_19"
Results_dir="results_video/${name}"
Base_dir="results_video"
Results_out="results_video/${name}/results"
# Write_results="results/quantitative/P4s/${name}"
device=2


CONFIG="models/Paint-by-Example/v5_Two_CLIP_proj_with_multiple_ID_losses/PBE/celebA/2024-02-27T00-54-33_v5_Two_CLIP_proj_with_multiple_ID_losses/configs/2024-02-27T00-54-33-project.yaml"
CKPT="models/Paint-by-Example/v5_Two_CLIP_proj_with_multiple_ID_losses/PBE/celebA/2024-02-27T00-54-33_v5_Two_CLIP_proj_with_multiple_ID_losses/checkpoints/last.ckpt"

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
    --target_video "examples/faceswap/News_reading.mp4" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --src_image "examples/faceswap/elon1.jpg" \
    --Base_dir "${Base_dir}" \
    --scale 3 \
    --ddim_steps 75 


    # --Start_from_target \
    # --target_start_noise_t 1000  
    

