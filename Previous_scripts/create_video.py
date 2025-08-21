import imageio
import os

def create_video_from_images(image_folder, output_path, fps=30):
    images = sorted([
        img for img in os.listdir(image_folder)
        if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")
    ])
    
    writer = imageio.get_writer(output_path, fps=fps)
    
    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        image = imageio.imread(img_path)
        writer.append_data(image)
    
    writer.close()
    print(f"Video saved to {output_path}")

# Example usage
create_video_from_images("/home/sanoojan/Paint_for_swap/results_video_new_REFace_analysis/vidcropped_face", "test_go_w_fl.mp4", fps=30)