device=3

CUDA_VISIBLE_DEVICES=${device} python scripts/inference_swap_video.py \
    --outdir "results_video_new/debug__def_start_no_feature_transfer" \
    --scale 3.5 \
    --ddim_steps 50 
