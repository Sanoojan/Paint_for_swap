current_time=$(date +"%Y%m%d_%H%M%S")
Write_results="/home/rusiru.achchige/Projects/REF/outputs/ref/Video_Quantitative"

# make folder if not exist
if [ ! -d "$Write_results" ]; then
    mkdir -p "$Write_results"
fi


output_filename="${Write_results}/out_${current_time}.txt"


source_path="dataset/FaceData/Data/VFHQ-Test/Celeb_Source"
Results_out="/home/rusiru.achchige/Projects/REF/outputs/ref/results_new/"
source_mask_path="/home/rusiru.achchige/Projects/REF/input_source_images/source_image_mask"
target_mask_path="/home/rusiru.achchige/Projects/REF/outputs/ref/results_video/"
# output_filename="/home/rusiru.achchige/Projects/REF/outputs/ref/Video_Quantitative/id_similarity.txt"
target_lables="/home/rusiru.achchige/Projects/REF/data/data_matching.yaml"
device=2


CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval_video .py --device cuda \
    "${source_path}" \
    "${Results_out}" \
    "${source_mask_path}" \
    "${target_mask_path}" \
    --dataset "ffhq" \
    --print_sim True  \
    --target_label_path $target_lables \
    --results_subfolder "model_outputs" \
    --target_subfolder "vidmask_frames" \
    --arcface True >> "$output_filename" 