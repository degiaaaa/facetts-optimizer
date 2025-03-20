import sys, os
import numpy as np
import subprocess
from glob import glob
import argparse

def extractData(videopath, audiopath):
    if audiopath is None:
        sys.exit('Audiopath must be provided.')

    audiocmd = f'ffmpeg -y -i "{videopath}" -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 "{audiopath}" -loglevel quiet'
    output = subprocess.call(audiocmd, shell=True, stdout=None)
    return output

def dataProcessing(videoroot, audioroot=None, listname=None):
    print('(INPUT)  VIDEO path: %s' % videoroot)
    print('(OUTPUT) AUDIO path: %s' % audioroot)

    # Handle the case where listname is None or the file doesn't exist
    if listname is None or not os.path.exists(listname):
        print(f"List file not found or not provided: {listname}")
        # Use glob to find video files recursively
        videolist = sorted(glob(videoroot + os.sep + '**' + os.sep + '*.mp4', recursive=True))
        print(f"Video files found: {len(videolist)}")
        prelist = []
    else:
        print(f'Read presaved list file from {listname}')
        with open(listname, 'r') as fid:
            prelist = fid.read().split('\n')[:-1]
        videolist = sorted([os.path.join(videoroot, videopath + '.mp4') for videopath in prelist])
        print(f"Videos listed in {listname}: {len(videolist)}")

    print('Number of files: %d' % len(videolist))

    for idx, filepath in enumerate(videolist):
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        # Construct audio output path using os.path functions
        if audioroot:
            # Get the relative path from the video root
            relative_path = os.path.relpath(filepath, videoroot)
            # Split the file path and replace the extension with .wav
            relative_path_no_ext, _ = os.path.splitext(relative_path)
            audioname = os.path.join(audioroot, f"{relative_path_no_ext}.wav")

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(audioname), exist_ok=True)


        # Extract audio
        extractData(filepath, audioname)

        sys.stdout.write('\rExtracting %s -- %03.03f%%' % (filepath, float(idx + 1) / len(videolist) * 100))
        sys.stdout.flush()

    sys.stdout.write('\n')

def main():
    parser = argparse.ArgumentParser(description='Extract audio from videos.')

    parser.add_argument('--vid_path', type=str, default='/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lrs2_splitted/mp4', help='Path to video files.')
    parser.add_argument('--aud_path', type=str, default='/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lrs2_splitted/wav', help='Path to save extracted audio files.')
    parser.add_argument('--list_path', type=str, help='Path to list file (optional).')

    args = parser.parse_args()

    print('Extract audio from videos.')
    dataProcessing(args.vid_path, args.aud_path, args.list_path)
    if args.list_path:
        print('Data list is processed from > %s' % args.list_path)
    print('Complete video extraction step!\n')

if __name__ == '__main__':
    main()
