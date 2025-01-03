import os
import shutil

# Pfade
syncnet_output_dir = "/mnt/qb/work2/butz1/bst080/syncnet_output"  # SyncNet-Ausgabeverzeichnis
input_txt_dir = "/mnt/qb/work2/butz1/bst080/spk_ids_main_weighted_40"  # Eingabeverzeichnis mit .txt-Dateien
facetts_output_dir = "/mnt/qb/work2/butz1/bst080/facetts_input"  # Ausgabe-Verzeichnis für FacetTS

# FacetTS-Verzeichnis erstellen
os.makedirs(facetts_output_dir, exist_ok=True)

# FacetTS-Dateien vorbereiten
def prepare_facetts_files(syncnet_output_dir, input_txt_dir, facetts_output_dir):
    for speaker_id in os.listdir(syncnet_output_dir):
        speaker_syncnet_dir = os.path.join(syncnet_output_dir, speaker_id)
        if not os.path.isdir(speaker_syncnet_dir):
            continue

        # FacetTS-Ausgabeverzeichnis für den Sprecher
        speaker_output_dir = os.path.join(facetts_output_dir, speaker_id)
        os.makedirs(speaker_output_dir, exist_ok=True)

        for video_folder in os.listdir(speaker_syncnet_dir):
            video_folder_path = os.path.join(speaker_syncnet_dir, video_folder)
            pyavi_path = os.path.join(video_folder_path, "pyavi", video_folder)

            # Konvertiere AVI zu MP4
            video_file = os.path.join(pyavi_path, "video.avi")
            if os.path.exists(video_file):
                mp4_file = os.path.join(speaker_output_dir, f"{video_folder}.mp4")
                convert_to_mp4(video_file, mp4_file)

            # Kopiere passende TXT-Datei
            txt_file = os.path.join(input_txt_dir, speaker_id, f"{video_folder}.txt")
            if os.path.exists(txt_file):
                shutil.copy(txt_file, os.path.join(speaker_output_dir, f"{video_folder}.txt"))

def convert_to_mp4(input_file, output_file):
    """
    Konvertiert .avi zu .mp4 mit FFmpeg.
    """
    command = f"ffmpeg -i {input_file} -c:v libx264 -preset slow -crf 22 -c:a aac -b:a 128k {output_file} -y"
    os.system(command)

# FacetTS-Dateien vorbereiten
prepare_facetts_files(syncnet_output_dir, input_txt_dir, facetts_output_dir)
print(f"FacetTS-compatible structure prepared at: {facetts_output_dir}")
