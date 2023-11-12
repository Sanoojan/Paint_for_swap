# CUDA_VISIBLE_DEVICES=2,3 python -u main_swap.py \
# --logdir models/Paint-by-Example/finetune/stable_diffusion/celebA/ \
# --pretrained_model pretrained_models/sd-v1-4-modified-9channel.ckpt \
# --base configs/v2.yaml \
# --scale_lr False


# CUDA_VISIBLE_DEVICES=2,3 python -u main_swap.py \
# --logdir models/Paint-by-Example/finetune/PBE/celebA/ \
# --pretrained_model checkpoints/model.ckpt \
# --base configs/v2.yaml \
# --scale_lr False

# CUDA_VISIBLE_DEVICES=0,5 python -u main_swap.py \
# --logdir models/Paint-by-Example/v9_add_feat/PBE/celebA/ \
# --pretrained_model checkpoints/model.ckpt \
# --base configs/v9_add_feat.yaml \
# --scale_lr False \
# --resume models/Paint-by-Example/v9_add_feat/PBE/celebA/2023-10-31T22-48-43_v9_add_feat/checkpoints/last.ckpt

# CUDA_VISIBLE_DEVICES=2,3 python -u main_swap.py \
# --logdir models/Paint-by-Example/v4_reconstruct_img_train_correct_gray_add_feature_2_final/PBE/celebA/ \
# --pretrained_model checkpoints/model.ckpt \
# --base configs/v4_reconstruct_img_train_correct.yaml \
# --scale_lr False \
# --resume models/Paint-by-Example/v4_reconstruct_img_train_correct_gray_add_feature_2_final/PBE/celebA/2023-11-02T13-17-29_v4_reconstruct_img_train_correct/checkpoints/last.ckpt


# CUDA_VISIBLE_DEVICES=2,3 python -u main_swap.py \
# --logdir models/Paint-by-Example/v4_reconstruct_img_train_correct_gray_add_feature_2/PBE/FFHQ/ \
# --pretrained_model checkpoints/model.ckpt \
# --base configs/v4_reconstruct_img_train_ffhq.yaml \
# --scale_lr False \
# --resume ?????


CUDA_VISIBLE_DEVICES=2,3 python -u main_swap.py \
--logdir models/Paint-by-Example/v4_reconstruct_img_train_normalize_clip_attention_diff_mask_augs/PBE/celebA/ \
--pretrained_model checkpoints/model.ckpt \
--base configs/v4_reconstruct_img_train_normalize_clip_attention_diff_mask_augs.yaml \
--scale_lr False 

# --resume models/Paint-by-Example/configs/v4_reconstruct_img_train_correct_normalize/PBE/celebA/2023-11-10T14-20-43_v4_reconstruct_img_train_correct_normalize/checkpoints/epoch=000002.ckpt


# --resume models/Paint-by-Example/v10_reconstruct_img_train_correct_attention_layers/PBE/celebA/2023-11-05T00-52-37_v10_reconstruct_img_train_correct_attention_layers/checkpoints/last.ckpt




# --resume models/Paint-by-Example/v4_reconstruct_img_train_correct_gray_add_feature/PBE/celebA/2023-10-29T23-59-03_v4_reconstruct_img_train_correct/checkpoints/last.ckpt




# --resume models/Paint-by-Example/v7_reconstruct_img_train_concat_feat/PBE/celebA/2023-10-27T00-54-41_v7_reconstruct_img_train_concat_feat/checkpoints/epoch=000012.ckpt