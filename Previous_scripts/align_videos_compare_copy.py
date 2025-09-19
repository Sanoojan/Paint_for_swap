import os
import yaml
from PIL import Image
from natsort import natsorted
from PIL import ImageFont, ImageDraw
import cv2

# ------------- unchanged definitions -------------
source_path = "dataset/FaceData/Data/VFHQ-Test/Celeb_Source"
target_folder = "/home/sanoojan/Paint_for_swap/dataset/FaceData/Data/VFHQ-Test/GT/Interval1_512x512_LANCZOS4/VIDEO"
Ours_folder   = "outputs/VFHQ_test_full/22_BGC_PnP_with_fft_feature_injection_all_with_op_flo_at_attn_trans_with_10_0.5/results_new/VIDEO/results"
Anyv2v_folder = "/home/sanoojan/Video_diffusion/AnyV2V/Results_full/Prompt-Based-Editing/i2vgen-xl/REFace1_new1/VIDEO"
GoWith_the_flow_folder = "/home/sanoojan/Go-with-the-Flow/results/VIDEO/images"
REFace_folder = "/home/sanoojan/Paint_for_swap/outputs/VFHQ_test_full/1_Vanilla_REFAce/results_new/VIDEO/results"
output_folder = "Plotting/combined_images_2_methods"
source_mapping_file = "outputs/VFHQ_test_full/1_REFace/results_new/data_matching.yaml"

# Align       = ['Source', 'Target', 'REFace', 'Anyv2v', 'GoWithTheFlow','VFace']
Align       = ['Source', 'Target', 'REFace','VFace']
# Video_names = ['Clip+1qf8dZpLED0+P2+C1+F5731-5855',
#                'Clip+2W7Bk7EcRMg+P0+C1+F3663-3770',
#                 'Clip+1L2d-mQA-Gc+P0+C0+F5084-5295',
#                 'Clip+5FRSvDOczJ8+P0+C0+F302-415',
#                 'Clip+KSF3tPr9zAk+P0+C2+F8769-8880',
#                 'Clip+oVkChb9DyoA+P0+C0+F5884-6070',
#                 'Clip+RfHY644z7aI+P0+C1+F2537-2828',
#                 'Clip+VjvX4tzzlbo+P2+C0+F5669-5935',
#                 'Clip+y59gJXBBXp0+P0+C0+F2590-2709']

Video_names = ['Clip+1qf8dZpLED0+P2+C1+F5731-5855',
                'Clip+2W7Bk7EcRMg+P0+C1+F3663-3770',
                'Clip+5FRSvDOczJ8+P0+C0+F302-415',
                'Clip+KSF3tPr9zAk+P0+C2+F8769-8880',
                'Clip+oVkChb9DyoA+P0+C0+F5884-6070',
                'Clip+RfHY644z7aI+P0+C1+F2537-2828',
                'Clip+y59gJXBBXp0+P0+C0+F2590-2709']

Video_names = ['Clip+KSF3tPr9zAk+P0+C2+F8769-8880']



os.makedirs(output_folder, exist_ok=True)

with open(source_mapping_file) as f:
    mapping = yaml.safe_load(f)

# ------------- NEW: build an index of sorted frame names -------------c 22
def get_dir(method, video):
    if   method == 'Target':        return target_folder.replace('VIDEO',       video)
    elif method == 'VFace':          return Ours_folder.replace('VIDEO',         video)
    elif method == 'Anyv2v':        return Anyv2v_folder.replace('VIDEO',       video)
    elif method == 'GoWithTheFlow': return GoWith_the_flow_folder.replace('VIDEO', video)
    elif method == 'REFace':      return REFace_folder.replace('VIDEO',       video)
    else:                           return None

frame_index = {}                       # frame_index[video][method] = [sorted list of files]
for video in Video_names:
    frame_index[video] = {}
    for method in Align:
        if method == 'Source':         # source uses the yaml mapping instead
            continue
        dir_path = get_dir(method, video)
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Folder missing: {dir_path}")
        files = [f for f in os.listdir(dir_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        frame_index[video][method] = natsorted(files)

# ------------- choose how many frames we can actually draw -------------
num_frames = 24   # you wanted 1-24
# sanity-check that every method has ≥ num_frames images
for video in Video_names:
    for method in Align[1:]:  # skip 'source'
        if len(frame_index[video][method]) < num_frames:
            raise ValueError(f"{method} for {video} has only "
                             f"{len(frame_index[video][method])} frames")

# ------------- make the combined grids -------------
# Try to load a clean sans-serif font; fallback if not available
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
except IOError:
    font = ImageFont.load_default()

for i in range(num_frames):            # i = 0 … 23
    grid_rows = []                     # one horizontal row per video

    for video in Video_names:
        method_imgs = []               # images for the five methods

        for method in Align:
            if method == 'Source':
                fname   = mapping[video]
                imgpath = os.path.join(source_path, fname)
            else:
                fname   = frame_index[video][method][i]          # i-th frame
                imgpath = os.path.join(get_dir(method, video), fname)

            if not os.path.exists(imgpath):
                print(f"[Warning] Missing {method} frame: {imgpath}")
                img = Image.new('RGB', (512, 512), (0, 0, 0))
            else:
                img = Image.open(imgpath).convert('RGB').resize((512, 512), Image.LANCZOS)
            method_imgs.append(img)

        # ▶ HORIZONTAL stack of the five methods
        w, h = method_imgs[0].size
        row_width  = w * len(method_imgs)
        row_canvas = Image.new('RGB', (row_width, h))
        for idx, im in enumerate(method_imgs):
            row_canvas.paste(im, (idx * w, 0))
        grid_rows.append(row_canvas)

    # ▶ VERTICAL stack of the two video rows
    total_w = grid_rows[0].width
    total_h = sum(im.height for im in grid_rows)

    final_grid = Image.new('RGB', (total_w, total_h))
    y_off = 0
    for im in grid_rows:
        final_grid.paste(im, (0, y_off))
        y_off += im.height

    # ▶ Add title row (above final_grid)
    title_height = 60
    full_img = Image.new('RGB', (total_w, total_h + title_height), (255, 255, 255))
    full_img.paste(final_grid, (0, title_height))

    # Draw text titles
    draw = ImageDraw.Draw(full_img)
    for idx, label in enumerate(Align):
        text_w, text_h = draw.textsize(label, font=font)
        x = idx * w + (w - text_w) // 2
        y = (title_height - text_h) // 2
        draw.text((x, y), label, fill='black', font=font)

    out_name = os.path.join(output_folder, f"frame_{i+1:06d}.jpg")
    full_img.save(out_name)
    print(f"Saved {out_name}")
    
    
# ----------- Create GIF from saved frames -----------
from PIL import Image

# Get sorted frame paths
frame_files = natsorted([
    os.path.join(output_folder, fname)
    for fname in os.listdir(output_folder)
    if fname.endswith(".jpg")
])

# Load all images
frames = [Image.open(f) for f in frame_files]

# Save as GIF
gif_path = os.path.join(output_folder, "comparison.gif")
frames[0].save(gif_path, format='GIF',
               save_all=True,
               append_images=frames[1:],
               duration=100,    # 100 ms per frame = 10 FPS
               loop=0)          # loop forever

print(f"GIF saved to: {gif_path}")


# Get sorted frame paths
frame_files = natsorted([
    os.path.join(output_folder, fname)
    for fname in os.listdir(output_folder)
    if fname.endswith(".jpg")
])

# Read first frame to get size
first_frame = cv2.imread(frame_files[0])
h, w, _ = first_frame.shape

# Define the codec and video writer
mp4_path = os.path.join(output_folder, "comparison.mp4")
fps = 10  # frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))

# Write each frame
for fpath in frame_files:
    frame = cv2.imread(fpath)
    out.write(frame)

out.release()
print(f"MP4 saved to: {mp4_path}")