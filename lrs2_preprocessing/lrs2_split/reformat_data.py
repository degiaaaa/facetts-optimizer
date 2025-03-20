import os
import shutil
import cv2

def load_filelist(filelist_path):
    """Load the file list from a given file."""
    filelist = set()
    with open(filelist_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            filelist.add(parts[0].strip())  # Remove unwanted spaces
    return filelist

def load_mapping(mapping_path):
    """Load the mapping file and store the mapping from old to new paths."""
    mapping = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' -> ')
            if len(parts) == 2 and parts[0].startswith("PROCESSED"):  # Only consider processed files
                old_path_full = parts[0].split(': ')[1].replace(".mp4", "")
                old_path = '/'.join(old_path_full.split('/')[-2:])  # Extract only the last two parts
                new_path = '/'.join(parts[1].split(' | ')[0].replace(".mp4", "").split('/')[-2:])
                mapping[old_path] = new_path
    return mapping

def get_video_duration(video_path):
    """Get the duration of a video file in seconds."""
    if not os.path.exists(video_path):
        return 0
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frame_count / fps
    return 0

def organize_files(src_root, dest_root, filelists, mapping):
    """Sort files based on the mapping and file lists."""
    splits = ['train', 'val', 'test']
    unmatched_dir = os.path.join(dest_root, "unmatched")
    too_short_dir = os.path.join(unmatched_dir, "too_short")
    os.makedirs(unmatched_dir, exist_ok=True)
    os.makedirs(too_short_dir, exist_ok=True)

    trainval_mp4_dir = os.path.join(dest_root, "mp4", "trainval")
    trainval_txt_dir = os.path.join(dest_root, "trainval")
    os.makedirs(trainval_mp4_dir, exist_ok=True)
    os.makedirs(trainval_txt_dir, exist_ok=True)
    
    for split, filelist in zip(splits, filelists):
        split_mp4_dir = os.path.join(dest_root, "mp4", split)
        split_txt_dir = os.path.join(dest_root, split)
        
        for old_path, new_path in mapping.items():
            old_path = old_path.strip()
            if old_path in filelist:
                path_parts = new_path.split('/')
                if len(path_parts) != 2:
                    print(f"Skipping malformed path: {new_path}")
                    continue
                
                speaker_id, filename = path_parts
                src_mp4 = os.path.join(src_root, new_path + ".mp4")
                src_txt = os.path.join(src_root, new_path + ".txt")
                dest_mp4_folder = os.path.join(split_mp4_dir, speaker_id)
                dest_txt_folder = os.path.join(split_txt_dir, speaker_id)

                if os.path.exists(src_mp4):
                    duration = get_video_duration(src_mp4)
                    if duration < 1.3:
                        short_mp4 = os.path.join(too_short_dir, new_path.replace('/', '_') + ".mp4")
                        short_txt = os.path.join(too_short_dir, new_path.replace('/', '_') + ".txt")
                        shutil.copy(src_mp4, short_mp4)
                        if os.path.exists(src_txt):
                            shutil.copy(src_txt, short_txt)
                        print(f"Moved short video: {src_mp4} -> {short_mp4}")
                        continue

                os.makedirs(dest_mp4_folder, exist_ok=True)
                os.makedirs(dest_txt_folder, exist_ok=True)

                dest_mp4 = os.path.join(dest_mp4_folder, filename + ".mp4")
                dest_txt = os.path.join(dest_txt_folder, filename + ".txt")
                dest_mp4_txt_folder = os.path.join(dest_txt_folder, filename + ".mp4")

                if os.path.exists(src_mp4):
                    shutil.copy(src_mp4, dest_mp4)
                    shutil.copy(src_mp4, dest_mp4_txt_folder)
                    print(f"Copied: {src_mp4} -> {dest_mp4} and {dest_mp4_txt_folder}")

                if os.path.exists(src_txt):
                    shutil.copy(src_txt, dest_txt)
                    print(f"Copied: {src_txt} -> {dest_txt}")

                # Also store in trainval if it's from train or val split
                if split in ["train", "val"]:
                    trainval_mp4_folder = os.path.join(trainval_mp4_dir, speaker_id)
                    trainval_txt_folder = os.path.join(trainval_txt_dir, speaker_id)
                    os.makedirs(trainval_mp4_folder, exist_ok=True)
                    os.makedirs(trainval_txt_folder, exist_ok=True)

                    trainval_mp4 = os.path.join(trainval_mp4_folder, filename + ".mp4")
                    trainval_txt = os.path.join(trainval_txt_folder, filename + ".txt")
                    trainval_mp4_txt_folder = os.path.join(trainval_txt_folder, filename + ".mp4")

                    if os.path.exists(src_mp4):
                        shutil.copy(src_mp4, trainval_mp4)
                        shutil.copy(src_mp4, trainval_mp4_txt_folder)
                        print(f"Copied to trainval: {src_mp4} -> {trainval_mp4} and {trainval_mp4_txt_folder}")

                    if os.path.exists(src_txt):
                        shutil.copy(src_txt, trainval_txt)
                        print(f"Copied to trainval: {src_txt} -> {trainval_txt}")

            else:
                unmatched_mp4 = os.path.join(unmatched_dir, new_path.replace('/', '_') + ".mp4")
                unmatched_txt = os.path.join(unmatched_dir, new_path.replace('/', '_') + ".txt")
                if os.path.exists(src_mp4):
                    shutil.copy(src_mp4, unmatched_mp4)
                if os.path.exists(src_txt):
                    shutil.copy(src_txt, unmatched_txt)
                print(f"WARNING: No filelist match for {old_path}, moving to unmatched")

# Pfade zu den benötigten Dateien
src_root = "/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lrs2_labeled_main"  # Quelle der neuen Dateien
dest_root = "/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lrs2_splitted"  # Zielort für den neuen Split
filelist_paths = [
    "/mnt/qb/work2/butz1/bst080/data/filelist_train",
    "/mnt/qb/work2/butz1/bst080/data/filelist_val",
    "/mnt/qb/work2/butz1/bst080/data/filelist_test"
]
mapping_path = "/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lookup_table_main.txt"  # Mapping-Datei

# Lade Dateilisten und Mapping
filelists = [load_filelist(path) for path in filelist_paths]
mapping = load_mapping(mapping_path)

# Organisiere die Dateien
organize_files(src_root, dest_root, filelists, mapping)

print("Dataset-Sortierung abgeschlossen!")
