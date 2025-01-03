import os
import shutil

def reformat_text_file(file_path):
    """Reformats the content of a text file."""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Extract the text content from the lines
        reformatted_content = ""
        for line in lines:
            if line.startswith("Text:"):
                reformatted_content += line.replace("Text:", "").strip() + " "

        # Convert to lowercase and remove extra spaces
        reformatted_content = reformatted_content.strip().lower()

        # Write the reformatted content back to the file
        with open(file_path, 'w') as file:
            file.write(reformatted_content)
        print(f"Reformatted: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def copy_and_process_folders(original_folder, copy_folder):
    """Copies the original folder to a new location and processes text files there."""
    try:
        # Copy the original folder to the new location
        if os.path.exists(copy_folder):
            print(f"Folder {copy_folder} already exists. Using existing folder.")
        else:
            shutil.copytree(original_folder, copy_folder)
            print(f"Copied {original_folder} to {copy_folder}.")

        # Process text files in the copied folder
        process_folders(copy_folder)
    except Exception as e:
        print(f"Error copying or processing folders: {e}")


def process_folders(root_folder):
    """Traverses all folders and processes text files."""
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(subdir, file)
                reformat_text_file(file_path)

if __name__ == "__main__":
    # Replace 'main_path' and 'copy_path' with your specific dataset paths
    main_path = "/mnt/qb/work2/butz1/bst080/spk_ids_main_weighted_40"
    copy_path = "/mnt/qb/work2/butz1/bst080/spk_ids_main_preprocessed"

    if os.path.exists(main_path):
        copy_and_process_folders(main_path, copy_path)
    else:
        print("The specified original folder path does not exist.")
