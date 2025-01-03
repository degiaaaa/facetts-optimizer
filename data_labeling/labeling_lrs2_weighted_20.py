import os
import cv2
import shutil
import numpy as np
import torch
from retinaface import RetinaFace
import face_recognition

# Check for GPU availability
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# Paths for dataset directories
base_path = "/mnt/qb/work2/butz1/bst080"
main_path = os.path.join(base_path, "working_copy_main")
speaker_ids_main = os.path.join(base_path, "spk_ids_main_weighted_20")

# Create the output folder if it doesn't exist
os.makedirs(speaker_ids_main, exist_ok=True)

# Function to check if a frame is blurry
def is_blurry(frame, threshold=20):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

# Function to extract multiple face encodings and confidence scores from a video using RetinaFace
def extract_faces_from_video_retinaface(video_path, sample_rate=5):
    video_capture = cv2.VideoCapture(video_path)
    face_encodings = []
    confidence_scores = []

    frame_count = 0
    success, frame = video_capture.read()
    while success:
        if frame_count % sample_rate == 0:  # Sample every `sample_rate` frames
            if not is_blurry(frame, threshold=20):  # Adjust blur detection threshold
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

        frame_count += 1
        success, frame = video_capture.read()

    video_capture.release()
    return face_encodings, confidence_scores

# Function to calculate a weighted average encoding
def weighted_average_encoding(encodings, weights=None):
    if not encodings:
        return None
    
    if weights is None:
        weights = np.ones(len(encodings))  # Equal weights by default

    weights = np.array(weights)
    return np.average(encodings, axis=0, weights=weights)

# Function to match a new encoding against known identities with verification
def match_faces(known_faces, face_encoding, threshold=0.5):
    distances = [np.linalg.norm(known_face - face_encoding) for known_face in known_faces]

    if distances:
        min_distance = min(distances)
        best_match_index = distances.index(min_distance)

        # Verification step
        if min_distance < threshold:
            return best_match_index, min_distance

    return -1, float("inf")

# Function to log ambiguous matches
def log_ambiguous_match(video_path, distances):
    ambiguous_log_path = os.path.join(base_path, "ambiguous_matches.log")
    with open(ambiguous_log_path, "a") as log_file:
        log_file.write(f"Ambiguous match for {video_path}: {distances}\n")

# Function to organize videos by detected face identity
def organize_videos_by_face_identity(path):
    known_faces = []  # Store average encodings for each known identity
    identity_count = 0  # Counter to create new identities
    processed_videos = set()  # Track processed videos to avoid duplicates

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                txt_path = video_path.replace(".mp4", ".txt")

                if video_path in processed_videos:
                    continue

                if not os.path.exists(txt_path):
                    print(f"Skipping {video_path} as the .txt file is missing.")
                    continue

                face_encodings, confidence_scores = extract_faces_from_video_retinaface(video_path, sample_rate=5)

                if not face_encodings:
                    print(f"No face detected in {video_path}. Skipping...")
                    continue

                # Use confidence scores as weights for the weighted average
                video_avg_encoding = weighted_average_encoding(face_encodings, weights=confidence_scores)

                identity_index, min_distance = match_faces(known_faces, video_avg_encoding, threshold=0.45)

                if identity_index == -1:
                    known_faces.append(video_avg_encoding)
                    identity_index = identity_count
                    identity_count += 1
                elif min_distance > 0.4:  # Ambiguity logging
                    log_ambiguous_match(video_path, [min_distance])

                identity_folder = os.path.join(speaker_ids_main, f"spk{identity_index:02}")
                os.makedirs(identity_folder, exist_ok=True)

                shutil.copy2(video_path, os.path.join(identity_folder, file))
                shutil.copy2(txt_path, os.path.join(identity_folder, os.path.basename(txt_path)))
                print(f"Stored {video_path} and {txt_path} in {identity_folder}")

                processed_videos.add(video_path)

# Organize videos in the specified main path
organize_videos_by_face_identity(main_path)
