
# Set variables
name="v5_old_man_video_test_inversion"
Results_dir="results_video/${name}"
Base_dir="results_video"
Results_out="results_video/${name}/results"
# Write_results="results/quantitative/P4s/${name}"
device=2


CONFIG="models/Paint-by-Example/v5_Two_CLIP_proj_154/checkpoints/project_ffhq.yaml"
CKPT="models/Paint-by-Example/V5_without_FSA_154/checkpoints/epoch=000019.ckpt"



current_time=$(date +"%Y%m%d_%H%M%S")
output_filename="${Write_results}/out_${current_time}.txt"



CUDA_VISIBLE_DEVICES=${device} python scripts/inference_swap_video.py \
    --outdir "${Results_dir}" \
    --target_video "examples/Video/An Old Man Doing Exercises For The Body And Mind.mp4" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --src_image "examples/FaceSwap_10/Source/elon.jpeg" \
    --Base_dir "${Base_dir}" \
    --scale 3 \
    --ddim_steps 30 \
    --Start_from_target 


    # --target_start_noise_t 1000  
    

