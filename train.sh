
# sleep 1h

# CUDA_VISIBLE_DEVICES=4,6 python -u main_swap.py \
# --logdir models/Paint-by-Example/v4_reconstruct_img_train_2_step_multi_false_from_stable_diff/PBE/celebA/ \
# --pretrained_model pretrained_models/sd-v1-4-modified-9channel.ckpt \
# --base configs/v4_reconstruct_img_train_2_step_multi_false.yaml \
# --scale_lr False 

CUDA_VISIBLE_DEVICES=0,1 python -u main_swap.py \
--logdir models/Paint-by-Example/v5_Two_CLIP_proj_with_multiple_ID_losses/PBE/celebA/ \
--pretrained_model checkpoints/model.ckpt \
--base configs/v5_Two_CLIP_proj_with_multiple_ID_losses.yaml \
--scale_lr False \
--resume models/Paint-by-Example/v5_Two_CLIP_proj_with_multiple_ID_losses/PBE/celebA/2024-02-27T00-54-33_v5_Two_CLIP_proj_with_multiple_ID_losses/checkpoints/epoch=000013.ckpt

# \
# --resume models/Paint-by-Example/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_noclip_same_image/PBE/celebA/2024-02-13T10-03-50_v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_noclip_same_image/checkpoints/last.ckpt

# \
# --resume models/Paint-by-Example/v4_reconstruct_SAME_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT/PBE/celebA/2024-02-01T17-18-39_v4_reconstruct_same_img_train_2_step_multi_false/checkpoints/last.ckpt

# \
# --resume models/Paint-by-Example/v4_reconstruct_img_train_2_step_multi_false_UN_NORM_CLIP_CORRECT/PBE/celebA/2024-01-24T13-41-55_v4_reconstruct_img_train_2_step_multi_false/checkpoints/epoch=000031.ckpt



# --resume models/Paint-by-Example/v4_reconstruct_img_train_4_step_multi_false_with_LPIPS/PBE/celebA/2024-01-02T07-50-43_v4_reconstruct_img_train_4_step_multi_false/checkpoints/epoch=000049.ckpt


# CUDA_VISIBLE_DEVICES=2,3 python -u main_swap.py \
# --logdir models/Paint-by-Example/v4_reconstruct_img_train_4_step_multi_false_with_LPIPS_train_attn_only/PBE/celebA/ \
# --pretrained_model checkpoints/model.ckpt \
# --base configs/v4_reconstruct_img_train_4_step_multi_false_with_LPIPS_train_attn_only.yaml \
# --scale_lr False \
# --resume models/Paint-by-Example/v4_reconstruct_img_train_4_step_multi_false_with_LPIPS_train_attn_only/PBE/celebA/2024-01-05T07-34-10_v4_reconstruct_img_train_4_step_multi_false_with_LPIPS_train_attn_only/checkpoints/last.ckpt
