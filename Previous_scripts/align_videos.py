from PIL import Image, ImageDraw, ImageFont
import os
from natsort import natsorted

# Paths
# source_path = "outputs/VFHQ_test_full/22_BGC_PnP_with_fft_feature_injection_all_with_op_flo_at_attn_trans_with_10_0.5/results_new/Clip+1qf8dZpLED0+P2+C1+F5731-5855/temp_results/1.png"
# target_folder = "outputs/VFHQ_test_full/22_BGC_PnP_with_fft_feature_injection_all_with_op_flo_at_attn_trans_with_10_0.5/results_video/Clip+1qf8dZpLED0+P2+C1+F5731-5855/vid"
# swapped_folder = "outputs/VFHQ_test_full/22_BGC_PnP_with_fft_feature_injection_all_with_op_flo_at_attn_trans_with_10_0.5/results_new/Clip+1qf8dZpLED0+P2+C1+F5731-5855/results"
# output_folder = "outputs/VFHQ_test_full/22_BGC_PnP_with_fft_feature_injection_all_with_op_flo_at_attn_trans_with_10_0.5/results_new/Clip+1qf8dZpLED0+P2+C1+F5731-5855/combined"

source_path = "outputs/VFHQ_test_full_multi/22_final_model/results_new/Clip+1qf8dZpLED0+P2+C1+F5731-5855/will_smith.jpeg/temp_results/will_smith.png"
target_folder = "outputs/VFHQ_test_full_multi/22_final_model/results_video/Clip+1qf8dZpLED0+P2+C1+F5731-5855/vid"
swapped_folder = "outputs/VFHQ_test_full_multi/22_final_model/results_new/Clip+1qf8dZpLED0+P2+C1+F5731-5855/will_smith.jpeg/results"
output_folder = "Plotting/teaser1"

source_path = "outputs/VFHQ_test_full_multi/22_final_model/results_new/Clip+y59gJXBBXp0+P0+C0+F2590-2709/selena.jpeg/temp_results/selena.png"
target_folder = "outputs/VFHQ_test_full_multi/22_final_model/results_video/Clip+y59gJXBBXp0+P0+C0+F2590-2709/vid"
swapped_folder = "outputs/VFHQ_test_full_multi/22_final_model/results_new/Clip+y59gJXBBXp0+P0+C0+F2590-2709/selena.jpeg/results"
output_folder = "Plotting/teaser1"

output_name="Clip+VjvX4tzzlbo+P2+C0+F5669-5935/elon.jpeg/vidtoelon_swap.mp4"


# Clip+VjvX4tzzlbo+P2+C0+F5669-5935/elon.jpeg/vidtoelon_swap.mp4


# Clip+1L2d-mQA-Gc+P0+C0+F5084-5295/3970-00.png/vidto3970-00_swap.mp4
# Clip+1L2d-mQA-Gc+P0+C0+F5084-5295/10076-00.png/vidto10076-00_swap.mp4


# Clip+1qf8dZpLED0+P2+C1+F5731-5855/yanlecun.jpeg/vidtoyanlecun_swap.mp4


# Clip+2W7Bk7EcRMg+P0+C1+F3663-3770/elon.jpeg/vidtoelon_swap.mp4


# Clip+5FRSvDOczJ8+P0+C0+F302-415/yanlecun.jpeg/vidtoyanlecun_swap.mp4


# Clip+6fU_dX14pk0+P1+C0+F3670-3774/selena.jpeg/vidtoselena_swap.mp4


# Clip+6SNHQfBQ6yk+P0+C1+F22194-22322/10783-00.png/vidto10783-00_swap.mp4
# Clip+6SNHQfBQ6yk+P0+C1+F22194-22322/elon.jpeg/vidtoelon_swap.mp4


# Clip+y59gJXBBXp0+P0+C0+F2590-2709/selena.jpeg/vidtoselena_swap.mp4


# Clip+x9klsDwglJM+P0+C1+F3802-3955/selena.jpeg/vidtoselena_swap.mp4


# Clip+WDN72QkW5KQ+P3+C0+F95232-95342/kevin.jpeg/vidtokevin_swap.mp4

selected_videos = [ "Clip+1qf8dZpLED0+P2+C1+F5731-5855/will_smith.jpeg/vidtowill_smith_swap.mp4",
                   "Clip+VjvX4tzzlbo+P2+C0+F5669-5935/elon.jpeg/vidtoelon_swap.mp4",
                    "Clip+1L2d-mQA-Gc+P0+C0+F5084-5295/3970-00.png/vidto3970-00_swap.mp4",
                    "Clip+1L2d-mQA-Gc+P0+C0+F5084-5295/10076-00.png/vidto10076-00_swap.mp4",
                    "Clip+1qf8dZpLED0+P2+C1+F5731-5855/yanlecun.jpeg/vidtoyanlecun_swap.mp4",
                    "Clip+2W7Bk7EcRMg+P0+C1+F3663-3770/elon.jpeg/vidtoelon_swap.mp4",
                    "Clip+5FRSvDOczJ8+P0+C0+F302-415/yanlecun.jpeg/vidtoyanlecun_swap.mp4",
                    "Clip+6fU_dX14pk0+P1+C0+F3670-3774/selena.jpeg/vidtoselena_swap.mp4",
                    "Clip+6SNHQfBQ6yk+P0+C1+F22194-22322/10783-00.png/vidto10783-00_swap.mp4",
                    "Clip+6SNHQfBQ6yk+P0+C1+F22194-22322/elon.jpeg/vidtoelon_swap.mp4",
                    "Clip+y59gJXBBXp0+P0+C0+F2590-2709/selena.jpeg/vidtoselena_swap.mp4",
                    "Clip+x9klsDwglJM+P0+C1+F3802-3955/selena.jpeg/vidtoselena_swap.mp4",
                    "Clip+WDN72QkW5KQ+P3+C0+F95232-95342/kevin.jpeg/vidtokevin_swap.mp4"]

for output_name in selected_videos:
    output_name= "Clip+1qf8dZpLED0+P2+C1+F5731-5855/will_smith.jpeg/vidtowill_smith_swap.mp4"
    All_source_path = "dataset/FaceData/Data/VFHQ-Test/Celebrities_source"

    All_swapped_path="outputs/VFHQ_test_full_multi/22_final_model/results_new"
    All_target_folder="outputs/VFHQ_test_full_multi/22_final_model/results_video"
    swapped_folder = os.path.join(All_swapped_path, output_name.split('/')[0], output_name.split('/')[1], 'results')
    source_name= output_name.split('/')[1]
    clip_name = output_name.split('/')[0]
    source_path = os.path.join(All_source_path, source_name)
    output_folder= os.path.join("Plotting", output_name.split('/')[0], output_name.split('/')[1]+ 'combined')
    target_folder = os.path.join(All_target_folder, clip_name, 'vid')

    os.makedirs(output_folder, exist_ok=True)

    # Load and resize source image
    source = Image.open(source_path).resize((256, 256))

    # Prepare drawing label on source
    source_draw = ImageDraw.Draw(source)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    source_draw.text((5, 5), "S", font=font, fill=(255, 255, 0))

    # Get sorted lists of filenames
    target_files = natsorted(os.listdir(target_folder))
    swapped_files = natsorted(os.listdir(swapped_folder))

    for i, (t_name, s_name) in enumerate(zip(target_files, swapped_files)):
        target = Image.open(os.path.join(target_folder, t_name)).resize((256, 256))
        swapped = Image.open(os.path.join(swapped_folder, s_name)).resize((512, 512))

        # Draw "T" on target
        target_draw = ImageDraw.Draw(target)
        target_draw.text((5, 5), "T", font=font, fill=(255, 255, 0))

        # Create a blank canvas
        combined = Image.new("RGB", (768, 512))
        draw_combined = ImageDraw.Draw(combined)

        # Paste images
        combined.paste(source, (0, 0))
        combined.paste(target, (0, 256))
        combined.paste(swapped, (256, 0))

        # Draw black vertical separator line at x=256
        draw_combined.line([(256, 0), (256, 512)], fill=(0, 0, 0), width=2)

        # Save combined image
        output_path = os.path.join(output_folder, f"combined_{i:02d}.jpg")
        combined.save(output_path)
        
        

    print("All combined images saved with 'S', 'T' labels and a separator line.")

    #save as mp4
    import cv2
    import numpy as np
    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for

    fps = 10  # Frames per second
    first_frame = cv2.imread(os.path.join(output_folder, "combined_00.jpg"))
    h, w, _ = first_frame.shape
    mp4_path = os.path.join(output_folder, "comparison.mp4")
    out = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
    # Write each frame
    for i in range(len(target_files)):
        frame_path = os.path.join(output_folder, f"combined_{i:02d}.jpg")
        frame = cv2.imread(frame_path)
        out.write(frame)    
        
    out.release()
    print(f"MP4 saved to: {mp4_path}")