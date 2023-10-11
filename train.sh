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
--logdir models/Paint-by-Example/v4_reconstruct_img_train_reducing_clip/PBE/celebA/ \
--pretrained_model checkpoints/model.ckpt \
--base configs/v4_reconstruct_img_train_reducing_clip.yaml \
--scale_lr False 