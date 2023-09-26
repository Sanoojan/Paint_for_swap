# CUDA_VISIBLE_DEVICES=2 python eval_tool/fid/fid_score.py --device cuda \
# dataset/FaceData/CelebAMask-HQ/Val \
# results/test_bench/results            # 14.029

# CUDA_VISIBLE_DEVICES=2 python eval_tool/fid/fid_score.py --device cuda \
# dataset/FaceData/CelebAMask-HQ/Val \
# /home/sanoojan/e4s/Results/testbench/results      #13.245

# CUDA_VISIBLE_DEVICES=2 python eval_tool/fid/fid_score.py --device cuda \
# dataset/FaceData/CelebAMask-HQ/Val \
# /home/sanoojan/e4s/Results/testbench/results_200000_celebA          #12.51

CUDA_VISIBLE_DEVICES=2 python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
dataset/FaceData/CelebAMask-HQ/Val \
/home/sanoojan/e4s/Results/testbench/results_200000_celebA          
# Top-1 accuracy: 19.49%
# Top-5 accuracy: 34.92%

# CUDA_VISIBLE_DEVICES=2 python eval_tool/ID_retrieval/ID_retrieval.py --device cuda \
# dataset/FaceData/CelebAMask-HQ/Val \
# results/test_bench/results        
# # Top-1 accuracy: 14.50%
# # Top-5 accuracy: 31.70%    

# CUDA_VISIBLE_DEVICES=2 python eval_tool/Pose/pose_compare.py --device cuda \
# dataset/FaceData/CelebAMask-HQ/Val_target \
# results/test_bench/results         # 5.59

# CUDA_VISIBLE_DEVICES=2 python eval_tool/Pose/pose_compare.py --device cuda \
# dataset/FaceData/CelebAMask-HQ/Val_target \
# /home/sanoojan/e4s/Results/testbench/results_200000_celebA          # 4.882


