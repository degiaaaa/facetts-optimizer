import os
import cv2

def analyze_videos(base_path):
    total_mp4_files = 0
    total_duration = 0.0
    short_videos = 0  # Videos < 1.3 seconds
    long_videos = 0   # Videos > 10 seconds
    
    # Traverse all subdirectories
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".mp4"):
                total_mp4_files += 1
                video_path = os.path.join(root, file)
                
                # Open video and calculate duration
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = total_frames / fps if fps > 0 else 0
                    cap.release()
                else:
                    print(f"Error opening file: {video_path}")
                    continue
                
                total_duration += duration
                
                # Count videos based on duration thresholds
                if duration < 1.3:
                    short_videos += 1
                elif duration > 10:
                    long_videos += 1
                
                print(f"File: {file}, Duration: {duration:.2f} seconds")
    
    # Calculate average duration
    average_duration = total_duration / total_mp4_files if total_mp4_files > 0 else 0
    
    # Print results
    print("\n=== Video Analysis Results ===")
    print(f"Total .mp4 Files: {total_mp4_files}")
    print(f"Average Duration: {average_duration:.2f} seconds")
    print(f"Files < 1.3 seconds: {short_videos}")
    print(f"Files > 10 seconds: {long_videos}")

# Path to the directory containing videos
base_path = "/mnt/qb/work2/butz1/bst080/spk_ids_main_weighted_40"

# Run the video analysis
analyze_videos(base_path)
