device=3

CUDA_VISIBLE_DEVICES=${device} python scripts/inference_swap_video.py \
    --outdir "results_video_new/elon/Only_spa_self_attn" \
    --scale 3.5 \
    --ddim_steps 50 \
    --n_frames 36 \
    --src_image "examples/FaceSwap_10/Source/elon.jpeg" 
