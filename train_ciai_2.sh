

python -u main_swap.py \
--logdir models/Paint-by-Example/SRC_CLIP_SRC_ID/PBE/FFHQ/ \
--pretrained_model checkpoints/model.ckpt \
--base configs/FFHQ/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS_src_conds.yaml \
--scale_lr False 
