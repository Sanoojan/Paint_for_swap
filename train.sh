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

CUDA_VISIBLE_DEVICES=2,3 python -u main_swap.py \
--logdir models/Paint-by-Example/finetune_with_Arcface_features_clip_avg/PBE/celebA/ \
--pretrained_model checkpoints/model.ckpt \
--base configs/v2.yaml \
--scale_lr False \
--resume "models/Paint-by-Example/finetune_with_Arcface_features_clip_avg/PBE/celebA/2023-09-18T22-18-23_v2/checkpoints/epoch=000009.ckpt" 