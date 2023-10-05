
# Set variables
Results_out="/home/sanoojan/e4s/Results/testbench/results_CelebA_ckpt_without_crop"
Write_results="results/quantitative/e4s/CelebA_ckpt_without_crop"

source_path="dataset/FaceData/CelebAMask-HQ/Val"
target_path="dataset/FaceData/CelebAMask-HQ/Val_target"


current_time=$(date +"%Y%m%d_%H%M%S")
output_filename="${Write_results}/out_${current_time}.txt"

if [ ! -d "$Write_results" ]; then
    mkdir -p "$Write_results"
    echo "Directory created: $Write_results"
else
    echo "Directory already exists: $Write_results"
fi


# Run FID score calculation
CUDA_VISIBLE_DEVICES=2 python eval_tool/fid/fid_score.py --device cuda \
    "${source_path}" \
    "${Results_out}/results"  >> "$output_filename"

CUDA_VISIBLE_DEVICES=2 python eval_tool/Pose/pose_compare.py --device cuda \
    "${target_path}" \
    "${Results_out}/results"  >> "$output_filename"
     
CUDA_VISIBLE_DEVICES=2 python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
    "${source_path}" \
    "${Results_out}/results"   >> "$output_filename"  

