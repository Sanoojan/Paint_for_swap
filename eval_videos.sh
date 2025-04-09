    device=0
    current_time=$(date +"%Y%m%d_%H%M%S")
    Write_results="Video_Quantitative"
    output_filename="${Write_results}/out_${current_time}.txt"

    target_path="/home/sanoojan/Video_diffusion/AnyV2V/data/Data/VFHQ-Test/GT/Interval1_512x512_LANCZOS4"
    Results_out="/home/sanoojan/Video_diffusion/AnyV2V/Results/outputs/ref/results_new"
    Results_out="/home/sanoojan/Video_diffusion/AnyV2V/Edited_frames/REFace"
    Results_out="/home/sanoojan/Video_diffusion/AnyV2V/Edited_frames/SimSwap"
    Results_out="/home/sanoojan/Video_diffusion/AnyV2V/Results_full/Prompt-Based-Editing/i2vgen-xl/REFace1"
    Results_out="/home/sanoojan/Video_diffusion/AnyV2V/Results_full/Prompt-Based-Editing/i2vgen-xl/SimSwap1"
    
    
    
    echo "Pose comarison with target:" >> "$output_filename"
    CUDA_VISIBLE_DEVICES=${device} python eval_tool/Pose/pose_compare.py --device cuda \
        "${target_path}" \
        "${Results_out}" \
        --vidfolders 10 \
        --batch-size 16 \
        --num_imgs 16 
        
        # >> "$output_filename"

    # echo "Expression comarison with target:" >> "$output_filename"
    # CUDA_VISIBLE_DEVICES=${device} python eval_tool/Expression/expression_compare_face_recon.py --device cuda \
    #     "${target_path}" \
    #     "${Results_out}" \
    #     --vidfolders 10 \
    #     --batch-size 16 \
    #     --num_imgs 16 
    #     # --subfolders "results"
        

        >> "$output_filename" \