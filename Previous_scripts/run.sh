
### Journal ###
root='results'
out_root='results/quantitative/metrics'


test_name='landmarks_with_neck'

test_image='clip_ID_landmark_avg_with_neck/results'
out_name=$test_name
need_post=1

CelebAHQ_GT='dataset/FaceData/CelebAMask-HQ/CelebA-HQ-img'

if [ ! -d "$out_root" ]; then
    mkdir -p "$out_root"
    echo "Directory created: $out_root"
else
    echo "Directory already exists: $out_root"
fi

CUDA_VISIBLE_DEVICES=3 python -u eval_tool/Restoreformer_metrics/cal_fid.py \
$root'/'$test_image \
--fid_stats 'experiments/pretrained_models/inception_FFHQ_512-f7b384ab.pth' \
--save_name $out_root'/'$out_name'_fid.txt' \

if [ -d $CelebAHQ_GT ]
then
    # PSRN SSIM LPIPS
    CUDA_VISIBLE_DEVICES=3 python -u eval_tool/Restoreformer_metrics/cal_psnr_ssim.py \
    $root'/'$test_image \
    --gt_folder $CelebAHQ_GT \
    --save_name $out_root'/'$out_name'_psnr_ssim_lpips.txt' \
    --need_post $need_post \

    # # # PSRN SSIM LPIPS
    CUDA_VISIBLE_DEVICES=3 python -u eval_tool/Restoreformer_metrics/cal_identity_distance.py  \
    $root'/'$test_image \
    --gt_folder $CelebAHQ_GT \
    --save_name $out_root'/'$out_name'_id.txt' \
    --need_post $need_post
else
    echo 'The path of GT does not exist'
fi