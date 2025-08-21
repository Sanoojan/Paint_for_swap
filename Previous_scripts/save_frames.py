import os
from natsort import natsorted
import cv2

Video_folders_path="dataset/FaceData/Data/test_videos"
save_frames_path="dataset/FaceData/Data/test_frames"

for video_folder in os.listdir(Video_folders_path):
    video_folder_path=os.path.join(Video_folders_path,video_folder)
    save_frames_folder_path=os.path.join(save_frames_path,video_folder)
    if not os.path.exists(save_frames_folder_path):
        os.makedirs(save_frames_folder_path)

    for video_file in os.listdir(video_folder_path):
        if video_file.endswith(".mp4") or video_file.endswith(".avi"):
            video_file_path=os.path.join(video_folder_path,video_file)
            output_folder = save_frames_folder_path
            os.makedirs(output_folder, exist_ok=True)

            video_capture = cv2.VideoCapture(video_file_path)
            frame_count = 0
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                # sample path 00000000.png
                # Save the frame as an image file
                frame_name = f"{frame_count:08d}.png"
                frame_path = os.path.join(output_folder, frame_name)
                # resize frame to 512
                frame = cv2.resize(frame, (512, 512))
                
                cv2.imwrite(frame_path, frame)
                frame_count += 1
            video_capture.release()
            print(f"Extracted {frame_count} frames from {video_file} and saved to {output_folder}")

