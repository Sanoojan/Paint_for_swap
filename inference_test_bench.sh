Results_out="results/test_bench_ori"

CUDA_VISIBLE_DEVICES=2 python scripts/inference_test_bench.py \
--plms \
--outdir ${Results_out} \
--config configs/v2.yaml \
--ckpt models/Paint-by-Example/finetune_with_Arcface_features_clip_avg/PBE/celbA/2023-09-18T22-18-23_v2/checkpoints/last.ckpt \
--scale 5

CUDA_VISIBLE_DEVICES=2 python eval_tool/fid/fid_score.py --device cuda \
dataset/FaceData/CelebAMask-HQ/Val \
${Results_out}/results