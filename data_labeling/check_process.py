import os
import cv2

# Paths for original and processed directories
original_path = "/mnt/qb/work2/butz1/bst080/working_copy_main"
processed_path = "/mnt/qb/work2/butz1/bst080/spk_ids_main_weighted_40"

# Function to check if a frame is blurry
def is_blurry(frame, threshold=40):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

# Function to collect all .mp4 files
def collect_mp4_files(path):
    files = []
    for root, dirs, file_list in os.walk(path):
        for file in file_list:
            if file.endswith(".mp4"):
                files.append(os.path.relpath(os.path.join(root, file), start=path))
    return files

# Analyze skipped files and reasons
def analyze_skipped_files(original_path, processed_path):
    original_files = set(collect_mp4_files(original_path))
    processed_files = set(collect_mp4_files(processed_path))

    # Files missing from the processed directory
    missing_files = original_files - processed_files

    reasons = {
        "blurry": [],
        "no_face_detected": [],
        "missing_txt_file": [],
        "other": []
    }

    for missing_file in missing_files:
        video_path = os.path.join(original_path, missing_file)
        txt_path = video_path.replace(".mp4", ".txt")

        if not os.path.exists(txt_path):
            reasons["missing_txt_file"].append(missing_file)
            continue

        video_capture = cv2.VideoCapture(video_path)
        success, frame = video_capture.read()
        blurry = False
        while success:
            if is_blurry(frame, threshold=40):
                blurry = True
                break
            success, frame = video_capture.read()
        video_capture.release()

        if blurry:
            reasons["blurry"].append(missing_file)
        else:
            reasons["no_face_detected"].append(missing_file)

    return reasons, len(processed_files)

# Count speaker folders in processed directory
def count_speaker_folders(processed_path):
    speaker_folders = [
        d for d in os.listdir(processed_path) if os.path.isdir(os.path.join(processed_path, d))
    ]
    return len(speaker_folders)

# Run analysis
reasons, total_processed_files = analyze_skipped_files(original_path, processed_path)
total_speakers = count_speaker_folders(processed_path)

# Output results
print("Summary of Processing:")
print(f"Total .mp4 files in the original directory: {len(collect_mp4_files(original_path))}")
print(f"Total .mp4 files in the processed directory: {total_processed_files}")
print(f"Total speakers detected (folders): {total_speakers}")
print("Summary of Skipped Files:")
for reason, files in reasons.items():
    print(f"Reason: {reason}, Count: {len(files)}")
    for file in files:
        print(f"  {file}")
