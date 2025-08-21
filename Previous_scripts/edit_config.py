import os
import glob

config_path="outputs/CelebVHQ_test_full/22_BGC_PnP_with_fft_feature_injection_all_with_op_flo_at_attn_trans_with_10_0.5/results_new/data_matching.yaml"
output_path="outputs/CelebVHQ_test_full/1_REFace/results_new"
edited_config_path="outputs/CelebVHQ_test_full/data_matching.yaml"

new_lines = []
with open(config_path, 'r') as f:
    lines = f.readlines()
    # breakpoint()
    for line in lines:
        vid = line.strip().split(': ')[0]
        output_vid_folder= glob.glob(os.path.join(output_path, vid,'*.mp4'))
        if len(output_vid_folder) > 0:
            new_lines.append(line)
            
with open(edited_config_path, 'w') as f:
    f.writelines(new_lines)
    print(f"Edited config saved to {edited_config_path}")