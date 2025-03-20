import os
import cv2

# Define paths
too_short_dir = "/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lrs2_splitted/unmatched/too_short"
unmatched_dir = "/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lrs2_splitted/unmatched"
mp4_root_dirs = [
    "/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lrs2_splitted/mp4/train",
    "/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lrs2_splitted/mp4/val",
    "/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lrs2_splitted/mp4/test"
]

def get_video_duration(video_path):
    """Get the duration of a video file in seconds."""
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frame_count / fps
    return None

def check_too_short_videos():
    """Check if all mp4 files in too_short are really under 1.3 seconds."""
    errors = []
    for file in os.listdir(too_short_dir):
        if file.endswith(".mp4"):
            file_path = os.path.join(too_short_dir, file)
            txt_path = os.path.join(too_short_dir, file.replace(".mp4", ".txt"))

            # Check if the corresponding txt file exists
            if not os.path.exists(txt_path):
                errors.append(f"ERROR: Missing TXT file for {file}")

            # Check video duration
            duration = get_video_duration(file_path)
            if duration is None:
                errors.append(f"ERROR: Could not read video {file}")
            elif duration >= 1.3:
                errors.append(f"ERROR: {file} should not be in too_short (Duration: {duration:.2f}s)")

    if errors:
        print("\n".join(errors))
    else:
        print("✅ All files in 'too_short' folder are correctly labeled.")

def check_unmatched_files():
    """Check if mp4 files in unmatched are actually not present in train/val/test."""
    errors = []
    files_to_remove = []

    for file in os.listdir(unmatched_dir):
        if file.endswith(".mp4"):
            file_path = os.path.join(unmatched_dir, file)
            txt_path = os.path.join(unmatched_dir, file.replace(".mp4", ".txt"))
            spk, vid_name = file.replace(".mp4", "").split("_", 1)

            # Check if the corresponding txt file exists
            if not os.path.exists(txt_path):
                errors.append(f"ERROR: Missing TXT file for {file}")

            # Check if the file exists in train/val/test
            found = False
            for root_dir in mp4_root_dirs:
                spk_dir = os.path.join(root_dir, spk)
                if os.path.exists(spk_dir):
                    possible_file = os.path.join(spk_dir, vid_name + ".mp4")
                    if os.path.exists(possible_file):
                        found = True
                        break  # Stop searching if found

            if found:
                print(f"🗑️ Removing {file} from unmatched (Found in {spk_dir})")
                files_to_remove.append(file_path)
                files_to_remove.append(txt_path)

    # Remove incorrect files from unmatched
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)

    if errors:
        print("\n".join(errors))
    else:
        print("✅ All files in 'unmatched' folder are correctly labeled.")

def remove_duplicates_from_unmatched():
    """Remove files from 'unmatched' if they are already in 'too_short'."""
    files_to_remove = []

    for file in os.listdir(unmatched_dir):
        if file.endswith(".mp4"):
            too_short_file = os.path.join(too_short_dir, file)
            unmatched_file = os.path.join(unmatched_dir, file)
            txt_file = unmatched_file.replace(".mp4", ".txt")

            # If the file exists in too_short, mark it for deletion
            if os.path.exists(too_short_file):
                print(f"🗑️ Removing duplicate {file} from unmatched (Already in too_short)")
                files_to_remove.append(unmatched_file)
                if os.path.exists(txt_file):
                    files_to_remove.append(txt_file)

    # Remove marked files
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)

    if files_to_remove:
        print("✅ Removed duplicate files from 'unmatched'.")
    else:
        print("✅ No duplicates found in 'unmatched'.")

if __name__ == "__main__":
    # print("🔍 Checking 'too_short' folder...")
    check_too_short_videos()

    # print("\n🔍 Checking 'unmatched' folder...")
    check_unmatched_files()

    print("\n🔍 Removing duplicates from 'unmatched' if they exist in 'too_short'...")
    remove_duplicates_from_unmatched()

    print("\n✅ Data validation complete.")
