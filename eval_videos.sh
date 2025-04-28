    device=0
    current_time=$(date +"%Y%m%d_%H%M%S")
    Write_results="Video_Quantitative"
    
    
    target_path="/home/sanoojan/Video_diffusion/AnyV2V/data/Data/VFHQ-Test/GT/Interval1_512x512_LANCZOS4"


    results_subfolder="results"
    
    Results_out="/home/sanoojan/Video_diffusion/AnyV2V/Results/outputs/ref/results_new"
    Results_out="/home/sanoojan/Video_diffusion/AnyV2V/Edited_frames/REFace"
    Results_out="/home/sanoojan/Video_diffusion/AnyV2V/Edited_frames/SimSwap"
    Results_out="/home/sanoojan/Video_diffusion/AnyV2V/Results_full/Prompt-Based-Editing/i2vgen-xl/REFace1"
    Results_out="/home/sanoojan/Video_diffusion/AnyV2V/Results_full/Prompt-Based-Editing/i2vgen-xl/SimSwap1"
    Results_out="outputs/VFHQ_test_full/12_PnP_with_feature_injection_0_6_tar_start/results_new"
    Results_out="outputs/VFHQ_test_full/11_PnP_with_feature_injection_for_inversion__tar_to_src_0_2_lyrs__avg_start/results_new"
    Results_out="outputs/VFHQ_test_full/13_PnP_with_feature_injection_0_6_adaIn_0_8_start_154_model/results_new"

    output_filename="${Write_results}/${Results_out}/out_${current_time}.txt"

    if [ ! -d "${Write_results}/${Results_out}" ]; then
        mkdir -p "${Write_results}/${Results_out}"
    fi

    echo "Video Quality Metrics:" > "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/common_metrics_on_video_quality/demo.py \
        --real_videos_path "${target_path}" \
        --generated_videos_path "${Results_out}" \
        --number_of_videos 10 \
        --video_length 16  > "$output_filename" \
        --only_final


    echo "Pose comarison with target:" >> "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/Pose/pose_compare.py --device cuda \
        "${target_path}" \
        "${Results_out}" \
        --vidfolders 10 \
        --batch-size 16 \
        --num_imgs 16 \
        --subfolders "${results_subfolder}" >> "$output_filename"

    echo "Expression comarison with target:" >> "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/Expression/expression_compare_face_recon.py --device cuda \
        "${target_path}" \
        "${Results_out}" \
        --vidfolders 10 \
        --batch-size 16 \
        --num_imgs 16 \
        --subfolders "${results_subfolder}" >> "$output_filename" 


    source_path="dataset/FaceData/Data/VFHQ-Test/Celeb_Source"
    # Results_out="outputs/VFHQ_test_full/12_PnP_with_feature_injection_0_6_tar_start/results_new"
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
        --results_subfolder "${results_subfolder}" >> "$output_filename"

    
    
    