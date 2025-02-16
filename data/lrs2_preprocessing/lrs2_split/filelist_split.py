import os

# Define paths
dest_root = "/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lrs2_splitted"
splits = ["train", "val", "test"]

# Output list files
output_files = {
    "train": os.path.join(dest_root, "datalist/lrs2_train_long.list"),
    "val": os.path.join(dest_root, "datalist/lrs2_val_long.list"),
    "test": os.path.join(dest_root, "datalist/lrs2_test_long.list"),
}

# Create datalist directory if it doesn't exist
os.makedirs(os.path.join(dest_root, "datalist"), exist_ok=True)

# Generate list files and count unique speakers
for split in splits:
    split_dir = os.path.join(dest_root, split)  # Example: /mnt/.../train
    output_file = output_files[split]
    
    # Collect all video file paths (without .mp4 extension)
    file_list = []
    for root, _, files in os.walk(split_dir):
        for file in files:
            if file.endswith(".mp4"):
                # Get relative path without extension
                relative_path = os.path.relpath(os.path.join(root, file), start=dest_root)
                relative_path = os.path.splitext(relative_path)[0]  # Remove .mp4 extension
                
                # Entferne das Präfix (split) aus dem relativen Pfad
                if relative_path.startswith(f"{split}/"):
                    relative_path = relative_path[len(f"{split}/"):]
                
                file_list.append(relative_path)
    
    # Write to the respective list file
    with open(output_file, "w") as f:
        for item in sorted(file_list):  # Ensure sorted order
            f.write(item + "\n")
    
    print(f"✅ Created {output_file} with {len(file_list)} entries.")
    
    # Count unique speakers
    unique_speakers = set()
    with open(output_file, "r") as f:
        for line in f:
            speaker_id = line.strip().split("/")[0]  # Extract speaker ID
            unique_speakers.add(speaker_id)
    
    print(f"🔹 Unique speakers in {split}: {len(unique_speakers)}")
