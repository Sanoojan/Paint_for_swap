
# Set variables
name="head_swap_check"
Results_dir="examples/FaceSwap_10/${name}/results"
Base_dir="examples/FaceSwap_10/${name}/Outs"
Results_out="examples/FaceSwap_10/${name}/results/results" 
# Write_results="results/quantitative/P4s/${name}"
device=3

# CONFIG="models/Paint-by-Example/v5_Two_CLIP_proj_154/checkpoints/project.yaml"
# CKPT="models/Paint-by-Example/v5_Two_CLIP_proj_154/checkpoints/last.ckpt"

CONFIG="models/Paint-by-Example/v5_SRC_CLIP_proj_with_multiple_ID_losses_random_masks/PBE/celebA/2024-02-28T21-01-11_v5_CLIP_proj_with_multiple_ID_losses/configs/2024-03-04T03-23-02-project.yaml"
CKPT="models/Paint-by-Example/v5_SRC_CLIP_proj_with_multiple_ID_losses_random_masks/PBE/celebA/2024-02-28T21-01-11_v5_CLIP_proj_with_multiple_ID_losses/checkpoints/epoch=000019.ckpt"

target_path="examples/FaceSwap_10/target_head"
source_path="examples/FaceSwap_10/source_head"


# current_time=$(date +"%Y%m%d_%H%M%S")
# output_filename="${Write_results}/out_${current_time}.txt"

# if [ ! -d "$Write_results" ]; then
#     mkdir -p "$Write_results"
#     echo "Directory created: $Write_results"
# else
#     echo "Directory already exists: $Write_results"
# fi

# Run inference

# ideal for small number iof samples

CUDA_VISIBLE_DEVICES=${device} python scripts/inference_swap_selected.py \
    --outdir "${Results_dir}" \
    --target_folder "${target_path}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --src_folder "${source_path}" \
    --Base_dir "${Base_dir}" \
    --n_samples 4 \
    --scale 3.5 \
    --ddim_steps 50



    # --Start_from_target \
    # --target_start_noise_t 1000  
    

