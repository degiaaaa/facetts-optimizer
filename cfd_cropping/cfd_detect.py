import os
import sys
# Add the path to the face-detection-pytorch directory
sys.path.append('facetts-optimizer/cfd_cropping/face-detection-pytorch')
import cv2
from detectors import DSFD
from PIL import Image
import numpy as np
import torch

# Ursprungsordner
source_dir = r'E:\cfd30norms\CFD Version 3.0\Images\CFD'

# Zielordner
target_dir = r'.\Chigago_rescaled_Images'

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Initialize the face detector
detector = DSFD(device='cpu')  # Use 'cuda' if you have a compatible GPU

def resize_with_aspect_ratio(image, target_size):
    img = Image.open(image)
    img.thumbnail(target_size)  # Resize image in place, maintaining aspect ratio
    return img

def resize_and_crop_face(image, target_size=(244, 244)):
    img_width, img_height = image.size
    aspect_ratio = img_width / img_height

    # Resize image
    if aspect_ratio > 1:  # Image is wider than tall
        new_width = int(aspect_ratio * target_size[1])
        image = image.resize((new_width, target_size[1]), Image.ANTIALIAS)
        left = (image.width - target_size[0]) // 2  # Center crop
        image = image.crop((left, 0, left + target_size[0], target_size[1]))
    else:  # Image is taller than wide or square
        new_height = int(target_size[0] / aspect_ratio)
        image = image.resize((target_size[0], new_height), Image.ANTIALIAS)
        top = (image.height - target_size[1]) // 2  # Center crop
        image = image.crop((0, top, target_size[0], top + target_size[1]))

    return image

# Iterate through all subdirectories and files in the source directory
for root, dirs, files in os.walk(source_dir):
    for idx, file in enumerate(files):
        if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):
            # Source file path
            source_file = os.path.join(root, file)

            # Resize the image while maintaining the aspect ratio
            img_resized = resize_with_aspect_ratio(source_file, (500, 500))  # Target size (width, height)

            # Convert to OpenCV format (for further OpenCV processing)
            img_resized_cv = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

            # Perform face detection using the resized image
            bboxes = detector.detect_faces(img_resized_cv, conf_th=0.9, scales=[0.5, 1])

            img_height, img_width, _ = img_resized_cv.shape  # Get image dimensions

            # Process each detected face
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2, _ = map(int, bbox)  # Convert to int for indexing

                # Crop the face
                face_img = img_resized_cv[y1:y2, x1:x2]

                # Convert the cropped face to PIL Image for resizing
                face_img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

                # Resize the cropped face to 244x244 without distortion
                #face_img_resized = resize_and_crop_face(face_img_pil, (244, 244))

                # Convert back to OpenCV format and save the cropped face as PNG
                #face_img_cv_resized = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
                target_file = os.path.join(target_dir, f"{os.path.splitext(file)[0]}_face{i}.png")
                cv2.imwrite(target_file, face_img)

                print(f"Image {file} face {i} saved as {target_file}.")
