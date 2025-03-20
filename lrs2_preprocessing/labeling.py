import os
import cv2
import shutil
import numpy as np
import torch
from retinaface import RetinaFace
import face_recognition

# Paths for dataset directories
base_path = "/mnt/qb/work2/butz1/bst080/data/mvlrs_v1"
context = "main"
main_path = os.path.join(base_path, context)
speaker_ids_main = os.path.join(base_path, f"lrs2_labeled_{context}")

# Create the output folder if it doesn't exist
os.makedirs(speaker_ids_main, exist_ok=True)

# Create folders for unprocessed videos
extra_folder = os.path.join(base_path, f"unprocessed_videos_{context}")
os.makedirs(extra_folder, exist_ok=True)

# Create subfolders for each reason why a file isn't in spk_ids_main
reasons = ["missing_txt", "no_face_detected", "no_encoding_extracted", "processing_error", "ambiguous_match"]
unprocessed_folders = {reason: os.path.join(extra_folder, reason) for reason in reasons}
for folder in unprocessed_folders.values():
    os.makedirs(folder, exist_ok=True)

# Function to check if a frame is blurry
def is_blurry(frame, threshold=40):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

# Function to extract multiple face encodings and confidence scores from a video using RetinaFace
def extract_faces_from_video_retinaface(video_path, sample_rate=5):
    video_capture = cv2.VideoCapture(video_path)
    face_encodings = []
    confidence_scores = []
    frame_qualities = []

    frame_count = 0
    success, frame = video_capture.read()
    while success:
        if frame_count % sample_rate == 0:  # Sample every `sample_rate` frames
            quality = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if not is_blurry(frame, threshold=40):  # Adjust blur detection threshold
                detections = RetinaFace.detect_faces(frame)  # Optimize RetinaFace usage

                if detections:
                    for key in detections:
                        face = detections[key]
                        facial_area = face['facial_area']
                        confidence = face['score']  # Get detection confidence

                        x1, y1, x2, y2 = facial_area
                        face_crop = frame[y1:y2, x1:x2]
                        rgb_face = cv2.resize(face_crop[:, :, ::-1], (128, 128))  # Resize to 128x128

                        # Get face encoding using face_recognition
                        encodings = face_recognition.face_encodings(rgb_face)
                        if encodings:
                            face_encodings.append(encodings[0])
                            confidence_scores.append(confidence)
                            frame_qualities.append(quality)

        frame_count += 1
        success, frame = video_capture.read()

    video_capture.release()
    return face_encodings, confidence_scores, frame_qualities

# Function to calculate a weighted average encoding
def weighted_average_encoding(encodings, confidences, qualities):
    if not encodings:
        return None

    # Combine confidence and quality to calculate weights
    combined_weights = np.array(confidences) * np.array(qualities)
    combined_weights = combined_weights / combined_weights.sum()  # Normalize weights

    return np.average(encodings, axis=0, weights=combined_weights)

# Function to match a new encoding against known identities with dynamic threshold adjustment and validation
def match_faces(known_faces, face_encoding, base_threshold=0.40):
    distances = [np.linalg.norm(known_face - face_encoding) for known_face in known_faces]

    if distances:
        min_distance = min(distances)
        avg_distance = np.mean(distances)
        dynamic_threshold = min(base_threshold, avg_distance * 0.8)  # Adjust threshold dynamically

        best_match_index = distances.index(min_distance)

        # Additional validation: Ensure match is significantly better than average
        if min_distance < dynamic_threshold:
            return best_match_index, min_distance

    return -1, float("inf")

# Function to reformat a text file
def reformat_text_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        reformatted_content = ""
        for line in lines:
            if line.startswith("Text:"):
                reformatted_content += line.replace("Text:", "").strip() + " "

        reformatted_content = reformatted_content.strip().lower()

        with open(file_path, 'w') as file:
            file.write(reformatted_content)
        print(f"Reformatted: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Function to log videos
def log_video(video_path, stored_path, lookup_table_path, reason=None, identity=None):
    with open(lookup_table_path, "a") as lookup_file:
        if reason:
            lookup_file.write(f"UNPROCESSED: {video_path} -> {stored_path} | Reason: {reason}\n")
        else:
            lookup_file.write(f"PROCESSED: {video_path} -> {stored_path} | Identity: {identity}\n")
    print(f"Logged: {video_path} -> {stored_path}")

def organize_videos_by_face_identity(main_path, base_path, context):
    known_faces = []
    identity_count = 0
    lookup_table_path = os.path.join(base_path, f"lookup_table_{context}.txt")

    with open(lookup_table_path, "w") as lookup_file:
        pass  # Clear the lookup table

    total_copied = 0
    total_no_faces = 0
    total_no_encoding = 0
    total_errors = 0

    for root, dirs, files in os.walk(main_path):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                txt_path = video_path.replace(".mp4", ".txt")

                try:
                    if not os.path.exists(txt_path):
                        reason = "missing_txt"
                        dest_path = os.path.join(unprocessed_folders[reason], os.path.basename(video_path))
                        shutil.copy2(video_path, dest_path)
                        log_video(video_path, dest_path, lookup_table_path, reason=reason)
                        continue

                    face_encodings, confidence_scores, frame_qualities = extract_faces_from_video_retinaface(video_path)
                    if not face_encodings:
                        reason = "no_face_detected"
                        dest_path = os.path.join(unprocessed_folders[reason], os.path.basename(video_path))
                        shutil.copy2(video_path, dest_path)
                        shutil.copy2(txt_path, dest_path.replace(".mp4", ".txt"))
                        log_video(video_path, dest_path, lookup_table_path, reason=reason)
                        total_no_faces += 1
                        continue

                    video_avg_encoding = weighted_average_encoding(face_encodings, confidence_scores, frame_qualities)
                    if video_avg_encoding is None:
                        reason = "no_encoding_extracted"
                        dest_path = os.path.join(unprocessed_folders[reason], os.path.basename(video_path))
                        shutil.copy2(video_path, dest_path)
                        shutil.copy2(txt_path, dest_path.replace(".mp4", ".txt"))
                        log_video(video_path, dest_path, lookup_table_path, reason=reason)
                        total_no_encoding += 1
                        continue

                    identity_index, min_distance = match_faces(known_faces, video_avg_encoding)
                    if identity_index == -1:
                        known_faces.append(video_avg_encoding)
                        identity_index = identity_count
                        identity_count += 1

                    identity_folder = os.path.join(speaker_ids_main, f"spk{identity_index:02}")
                    os.makedirs(identity_folder, exist_ok=True)

                    stored_video_path = os.path.join(identity_folder, file)
                    shutil.copy2(video_path, stored_video_path)

                    stored_txt_path = os.path.join(identity_folder, os.path.basename(txt_path))
                    shutil.copy2(txt_path, stored_txt_path)
                    reformat_text_file(stored_txt_path)

                    log_video(video_path, stored_video_path, lookup_table_path, identity=f"spk{identity_index:02}")
                    total_copied += 1

                except Exception as e:
                    reason = "processing_error"
                    dest_path = os.path.join(unprocessed_folders[reason], os.path.basename(video_path))
                    shutil.copy2(video_path, dest_path)
                    if os.path.exists(txt_path):
                        shutil.copy2(txt_path, dest_path.replace(".mp4", ".txt"))
                    log_video(video_path, dest_path, lookup_table_path, reason=reason)
                    total_errors += 1
                    print(f"Error processing video {video_path}: {e}")

    print("\n=== Summary ===")
    print(f"Total Videos Copied: {total_copied}")
    print(f"Total Videos Skipped (No Faces): {total_no_faces}")
    print(f"Total Videos Skipped (No Encoding): {total_no_encoding}")
    print(f"Total Errors During Processing: {total_errors}")

# Run the organization function
organize_videos_by_face_identity(main_path, base_path, context)
