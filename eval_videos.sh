    device=1
    current_time=$(date +"%Y%m%d_%H%M%S")
    Write_results="Video_Quantitative2"
    
    
    target_path="dataset/FaceData/Data/VFHQ-Test/GT/Interval1_512x512_LANCZOS4"
    crop_coordinates="/home/sanoojan/Paint_for_swap/outputs/VFHQ_test_full/1_REFace/results_video"
    target_lables="dataset/FaceData/Data/VFHQ-Test/data_matching.yaml"
    target_lables="dataset/FaceData/Data/VFHQ-Test/data_matching_simswap.yaml"
    source_path="dataset/FaceData/Data/VFHQ-Test/Celeb_Source"
    source_mask_path="input_source_images/source_image_mask"
    target_mask_path="outputs/VFHQ_test_full/22_BGC_PnP_with_fft_feature_injection_all_with_op_flo_at_attn_trans_with_10_0.5/results_video"

    #################### /Change here ####################
    
    ## No sub folder
    # Results_out="/home/sanoojan/Video_diffusion/AnyV2V/Results/outputs/ref/results_new"
    # Results_out="/home/sanoojan/Video_diffusion/AnyV2V/Edited_frames/REFace"
    # Results_out="/home/sanoojan/Video_diffusion/AnyV2V/Edited_frames/SimSwap2"
    # Video_results_out="/home/sanoojan/Video_diffusion/AnyV2V/Edited_frames/SimSwap_videos2"
    # # Results_out=""/home/sanoojan/Video_diffusion/AnyV2V/data/Data/VFHQ-Test/GT/Interval1_512x512_LANCZOS4""

    # results_subfolder=""


    ## Results in sub folder
    Results_out="outputs/VFHQ_test_full/13_PnP_with_feature_injection_0_6_adaIn_0_8_start/results_new"
    Results_out="outputs/VFHQ_test_full/17_PnP_with_feature_injection_fft_3_1_gaussian_temporal_smoothneing/results_new"
    Results_out="outputs/VFHQ_test_full/1_REFace/results_new"
    Results_out="outputs/VFHQ_test_full/19_PnP_with_fft_feature_injection/results_new"
    Results_out="outputs/VFHQ_test_full/6_PnP_on_REFace/results_new"
    Results_out="outputs/VFHQ_test_full/14_PnP_with_feature_injection_154_fft_3_1/results_new"
    Results_out="outputs/VFHQ_test_full/20_PnP_with_fft_feature_injection_all_with_op_flo_img_141/results_new"
    Results_out="outputs/VFHQ_test_full/22_BGC_PnP_with_fft_feature_injection_all_with_op_flo_at_attn_trans_with_10_0.5/results_new"
    # Results_out="outputs/VFHQ_test_full/1_Vanilla_REFAce/results_new"
    # Results_out="/home/sanoojan/Video_diffusion/AnyV2V/Results_full/Prompt-Based-Editing/i2vgen-xl/REFace1_new1"
    # Results_out="/home/sanoojan/Go-with-the-Flow/results"
    Results_out="outputs/VFHQ_test_full/23_Final_fft0.6/results_new"
    # Results_out="outputs/VFHQ_test_full/23_only_FATS_TSG/results_new"
    # Results_out="outputs/VFHQ_test_full/23_Final_fft0.4/results_new"
    Results_out="outputs/VFHQ_test_full/23_Final_fft0.8_alpha0.8_steps40/results_new"
    Video_results_out="outputs/VFHQ_test_full/23_Final_fft0.8_alpha0.8_steps40/results_new"
    
    # Video_results_out="outputs/VFHQ_test_full/1_REFace/results_new"

    # Video_results_out="/home/sanoojan/Go-with-the-Flow/results"
    results_subfolder="results"

    #########################################################################################

    output_filename="${Write_results}/${Results_out}/out_${current_time}.txt"

    if [ ! -d "${Write_results}/${Results_out}" ]; then
        mkdir -p "${Write_results}/${Results_out}"
    fi

    echo "Video Quality Metrics:" > "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/common_metrics_on_video_quality/demo.py \
        --real_videos_path "${target_path}" \
        --generated_videos_path "${Results_out}" \
        --generated_subfolder "${results_subfolder}" \
        --number_of_videos -1 \
        --video_length 24 \
        --video_names_details ${target_lables} \
        --only_final >> "$output_filename"

    CUDA_VISIBLE_DEVICES=${device} python eval_tool/content-debiased-fvd/test.py \
        --results "${Video_results_out}" >> "$output_filename"



    echo "Pose comarison with target:" >> "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/Pose/pose_compare.py --device cuda \
        "${target_path}" \
        "${Results_out}" \
        --vidfolders -1 \
        --batch-size 24 \
        --video_names_details $target_lables \
        --num_imgs 24 \
        --crop_coordinates "${crop_coordinates}" \
        --subfolders "${results_subfolder}" >> "$output_filename"



    echo "Expression comarison with target:" >> "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/Expression/expression_compare_face_recon.py --device cuda \
        "${target_path}" \
        "${Results_out}" \
        --vidfolders -1 \
        --batch-size 24 \
        --num_imgs 24 \
        --video_names_details $target_lables \
        --crop_coordinates "${crop_coordinates}" \
        --subfolders "${results_subfolder}" >> "$output_filename" 


    echo "ID retrieval:" >> "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/ID_retrieval/ID_retrieval_video.py --device cuda \
        "${source_path}" \
        "${Results_out}" \
        "${source_mask_path}" \
        "${target_mask_path}" \
        --dataset "ffhq" \
        --print_sim True  \
        --target_label_path $target_lables \
        --target_subfolder "vidmask_frames" \
        --arcface True \
        --number_of_images 24 \
        --crop_coordinates "${crop_coordinates}" \
        --results_subfolder "${results_subfolder}"  >> "$output_filename" 
        
      