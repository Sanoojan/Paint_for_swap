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