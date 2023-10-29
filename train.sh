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
--logdir models/Paint-by-Example/v7_reconstruct_img_train_concat_feat/PBE/celebA/ \
--pretrained_model checkpoints/model.ckpt \
--base configs/v7_reconstruct_img_train_concat_feat.yaml \
--scale_lr False \
--resume models/Paint-by-Example/v7_reconstruct_img_train_concat_feat/PBE/celebA/2023-10-27T00-54-41_v7_reconstruct_img_train_concat_feat/checkpoints/epoch=000012.ckpt