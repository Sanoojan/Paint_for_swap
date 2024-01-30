# python scripts/inference.py \
# --plms --outdir results \
# --config configs/v1.yaml \
# --ckpt checkpoints/model.ckpt \
# --image_path examples/image/example_1.png \
# --mask_path examples/mask/example_1.png \
# --reference_path examples/reference/example_1.jpg \
# --seed 321 \
# --scale 5

# python scripts/inference.py \
# --plms --outdir results \
# --config configs/v1.yaml \
# --ckpt checkpoints/model.ckpt \
# --image_path examples/image/example_2.png \
# --mask_path examples/mask/example_2.png \
# --reference_path examples/reference/example_2.jpg \
# --seed 5876 \
# --scale 5

# CUDA_VISIBLE_DEVICES=2 python scripts/inference.py \
# --plms --outdir results \
# --config configs/v1.yaml \
# --ckpt checkpoints/model.ckpt \
# --image_path examples/image/example_3.png \
# --mask_path examples/mask/example_3.png \
# --reference_path examples/reference/example_3.jpg \
# --seed 5065 \
# --scale 5

# CUDA_VISIBLE_DEVICES=2 python scripts/inference.py \
# --plms --outdir results \
# --config configs/v1.yaml \
# --ckpt models/Paint-by-Example/finetune/PBE/celebA/2023-09-13T22-14-47_v2/checkpoints/last.ckpt \
# --image_path /home/sanoojan/e4s/example/input/faceswap/andy/546.jpg \
# --mask_path /home/sanoojan/e4s/data/FaceData/CelebAMask-HQ/CelebA-HQ-mask/0/00000_skin.png \
# --reference_path /home/sanoojan/e4s/example/input/faceswap/peng/peng_800.jpg \
# --seed 5065 \
# --scale 5

CUDA_VISIBLE_DEVICES=2 python scripts/inference.py \
--n_imgs 500 \
--plms --outdir results/arcface_clip_avg \
--config configs/v1.yaml \
--ckpt models/Paint-by-Example/finetune_with_Arcface_features_clip_avg/PBE/celbA/2023-09-18T22-18-23_v2/checkpoints/last.ckpt \
--image_path /home/sanoojan/e4s/data/FaceData/CelebAMask-HQ/CelebA-HQ-img/28002.jpg \
--mask_path /home/sanoojan/e4s/data/FaceData/CelebAMask-HQ/CelebA-HQ-mask/14/28002_skin.png \
--reference_path /home/sanoojan/e4s/data/FaceData/CelebAMask-HQ/CelebA-HQ-img/28003.jpg \
--seed 5065 \
--scale 5
