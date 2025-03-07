import os
import sys

# Add path to config.py
sys.path.append("/mnt/qb/work/butz/bst080/faceGANtts/")
import glob

from config import config  # Import config function

def count_speakers_and_videos():
    # Load dataset paths from config
    config_ = config()

    # Dataset paths
    lrs3_train = config_['lrs3_train']
    lrs3_val = config_['lrs3_val']
    lrs3_test = config_['lrs3_test']
    lrs2_splitted = config_['lrs3_path']

    dataset_files = {
        "train": lrs3_train,
        "val": lrs3_val,
        "test": lrs3_test
    }

    total_unique_speakers = set()
    speakers_per_path = {}
    videos_per_path = {}

    for name, file_path in dataset_files.items():
        unique_speakers = set()
        total_videos = 0

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        speaker_id = line.split('/')[0]  # Extract "spkXX"
                        unique_speakers.add(speaker_id)
                        total_videos += 1  # Each line represents one MP4 video

            speakers_per_path[name] = len(unique_speakers)
            videos_per_path[name] = total_videos
            total_unique_speakers.update(unique_speakers)

        except FileNotFoundError:
            print(f"[WARNING] File not found: {file_path}")
            speakers_per_path[name] = 0
            videos_per_path[name] = 0

    # Print results
    print(f"\n[INFO] Total unique speakers across all datasets: {len(total_unique_speakers)}")
    print(f"[INFO] Total videos across all datasets: {sum(videos_per_path.values())}\n")
    
    for dataset in dataset_files.keys():
        print(f"[INFO] Unique speakers in {dataset}: {speakers_per_path[dataset]}")
        print(f"[INFO] Total videos in {dataset}: {videos_per_path[dataset]}\n")

    # -----------------------
    # NEW: Count words in train, val, and test directories
    # -----------------------
    def count_words_in_directory(directory_path):
        """
        Counts the total number of space-separated words in all .txt files
        within the specified directory (recursively).
        """
        total_words = 0
        for root, dirs, files in os.walk(directory_path):  # Reference 1
            for filename in files:
                if filename.endswith(".txt"):
                    file_path = os.path.join(root, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:  # Reference 2
                        text = f.read()
                        words = text.split()  # Reference 3
                        total_words += len(words)
        return total_words

    if lrs2_splitted:
        train_dir = os.path.join(lrs2_splitted, "train")
        val_dir   = os.path.join(lrs2_splitted, "val")
        test_dir  = os.path.join(lrs2_splitted, "test")

        print("\n[INFO] Counting words in each split directory...")

        train_word_count = count_words_in_directory(train_dir)
        val_word_count   = count_words_in_directory(val_dir)
        test_word_count  = count_words_in_directory(test_dir)

        print(f"[INFO] Total words in train directory: {train_word_count}")
        print(f"[INFO] Total words in val directory:   {val_word_count}")
        print(f"[INFO] Total words in test directory:  {test_word_count}")
    else:
        print("\n[WARNING] 'lrs2_splitted' environment variable/path is not set. Word counting skipped.")

if __name__ == "__main__":
    count_speakers_and_videos()
