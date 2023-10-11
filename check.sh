CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path dataset/FaceData/CelebAMask-HQ/Val_target/28186.jpg \
--mask_path dataset/FaceData/CelebAMask-HQ/CelebA-HQ-mask/14/28186_skin.png \
--reference_path dataset/FaceData/CelebAMask-HQ/Val/29186.jpg \
--seed 321 \
--scale 5