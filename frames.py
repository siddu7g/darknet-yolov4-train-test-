import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=9):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done. Extracted {saved_count} images from {video_path}.")

if __name__ == "__main__":
    # === CHANGE THESE ===
    video_path = "/home/sidg/RecFile_1_20250613_165924_yuv_2_rgb_camera_back_1_output.avi"    # Path to your .avi file
    output_dir = "/home/sidg/Delta/all_data"        # Folder to save images

    extract_frames(video_path, output_dir)

