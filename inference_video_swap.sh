
# Set variables
name="New_v4_no_grad_ep49_mouth_p"
Results_dir="results_video/${name}"
Results_out="results_video/${name}/results"
# Write_results="results/quantitative/P4s/${name}"
device=1

CONFIG="models/Paint-by-Example_no_grad/v4_reconstruct_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT/PBE/celebA/2024-01-24T13-41-55_v4_reconstruct_img_train_2_step_multi_false/configs/2024-01-24T13-41-55-project.yaml"
CKPT="models/Paint-by-Example_no_grad/v4_reconstruct_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT/PBE/celebA/2024-01-24T13-41-55_v4_reconstruct_img_train_2_step_multi_false/checkpoints/epoch=000049.ckpt"

source_path="dataset/FaceData/CelebAMask-HQ/Val"
source_mask_path="dataset/FaceData/CelebAMask-HQ/src_mask"


current_time=$(date +"%Y%m%d_%H%M%S")
output_filename="${Write_results}/out_${current_time}.txt"

if [ ! -d "$Write_results" ]; then
    mkdir -p "$Write_results"
    echo "Directory created: $Write_results"
else
    echo "Directory already exists: $Write_results"
fi

# Run inference

CUDA_VISIBLE_DEVICES=${device} python scripts/inference_swap_video.py \
    --outdir "${Results_dir}" \
    --target_video "examples/faceswap/Andy2.mp4" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --src_image "examples/faceswap/target.jpg" \
    --scale 5 


    # --Start_from_target \
    # --target_start_noise_t 1000  
    

