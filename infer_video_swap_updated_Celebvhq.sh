
##### EXPERIMENTAL #####

Base_dir="outputs/CelebVHQ_test_full"
Experiment_name="22_BGC_PnP_with_fft_feature_injection_all_with_op_flo_at_attn_trans_with_10_0.5_new_test"
device=3

CONFIG="models/Paint-by-Example/v5_Two_CLIP_proj_154/checkpoints/project_ffhq.yaml"
CKPT="models/Paint-by-Example/v5_Two_CLIP_proj_154/checkpoints/last.ckpt"
# CKPT="models/Paint-by-Example/V5_without_FSA_154/checkpoints/epoch=000019.ckpt"

video_base_dir="dataset/FaceData/Data/test_videos"
image_dir="/home/sanoojan/Paint_for_swap/dataset/FaceData/Data/VFHQ-Test/Celeb_Source_100"
DATA_CONFIG="${Base_dir}/${Experiment_name}/results_new/data_matching.yaml"

if [ ! -d "${Base_dir}/${Experiment_name}/results_new" ]; then
    mkdir -p "${Base_dir}/${Experiment_name}/results_new"
fi

current_time=$(date +"%Y%m%d_%H%M%S")

python generate_config.py \
    --video_base_dir "${video_base_dir}" \
    --image_dir "${image_dir}" \
    --output_yaml_path "${DATA_CONFIG}"


CUDA_VISIBLE_DEVICES=${device} python scripts/inference_video.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --data_config "${DATA_CONFIG}" \
    --Base_dir "${Base_dir}/${Experiment_name}/results_video" \
    --video_base_dir "${video_base_dir}" \
    --image_dir "${image_dir}" \
    --output_base_dir "${Base_dir}/${Experiment_name}/results_new" \
    --scale 3.5 \
    --ddim_steps 50 

    

