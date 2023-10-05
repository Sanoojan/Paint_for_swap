CUDA_VISIBLE_DEVICES=2 python scripts/inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/example_2.png \
--mask_path examples/mask/example_2.png \
--reference_path examples/reference/example_2.jpg \
--seed 321 \
--scale 5