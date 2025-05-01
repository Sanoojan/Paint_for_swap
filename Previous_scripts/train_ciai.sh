


python -u main_swap.py \
--logdir models/Paint-by-Example/Target_CLIP_SRC_ID/PBE/FFHQ/ \
--pretrained_model checkpoints/model.ckpt \
--base configs/FFHQ/v4_reconstruct_img_train_2_step_multi_false_with_LPIPS.yaml \
--scale_lr False 
