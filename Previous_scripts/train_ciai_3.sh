

python -u main_swap.py \
--logdir /l/users/muhammad.haris/Sanoojan/Outputs/models/Paint-by-Example/TAR_CLIP_SRC_ID_No_aug_target/PBE/FFHQ/ \
--pretrained_model checkpoints/model.ckpt \
--base configs/FFHQ/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS.yaml \
--scale_lr False \
--resume /l/users/muhammad.haris/Sanoojan/Outputs/models/Paint-by-Example/Target_CLIP_SRC_ID/PBE/FFHQ/2024-02-16T23-11-43_v4_reconstruct_img_train_2_step_multi_false_with_LPIPS/checkpoints/epoch=000009.ckpt

