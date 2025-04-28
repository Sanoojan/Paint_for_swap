current_time=$(date +"%Y%m%d_%H%M%S")
Write_results="Video_Quantitative"

# make folder if not exist
if [ ! -d "$Write_results" ]; then
    mkdir -p "$Write_results"
fi


output_filename="${Write_results}/out_${current_time}.txt"


source_path="dataset/FaceData/Data/VFHQ-Test/Celeb_Source"
Results_out="outputs/VFHQ_test_full/12_PnP_with_feature_injection_0_6_tar_start/results_new"
source_mask_path="input_source_images/source_image_mask"
target_mask_path="outputs/VFHQ_test_full/11_PnP_with_feature_injection_for_inversion__tar_to_src_0_2_lyrs__avg_start/results_video"
target_lables="dataset/FaceData/Data/VFHQ-Test/data_matching.yaml"
device=0


CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval_video.py --device cuda \
    "${source_path}" \
    "${Results_out}" \
    "${source_mask_path}" \
    "${target_mask_path}" \
    --dataset "VFHQ" \
    --print_sim True  \
    --target_label_path $target_lables \
    --target_subfolder "vidmask_frames" \
    --arcface True \
    --number_of_images 16 \
    --results_subfolder "results" 
    
    # >> "$output_filename" 

        