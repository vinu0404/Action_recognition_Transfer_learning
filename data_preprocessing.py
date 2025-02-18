import glob
import cv2
import numpy as np
import random
import tensorflow as tf
from config import CLASSES, DATASET_PATH

def format_frames(frame, output_size=(224, 224)):
    frame = tf.keras.layers.Rescaling(1.0 / 255.0)(frame)  # Normalize
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def frames_from_video_file(video_path, n_frames=10, output_size=(224, 224), frame_step=15):
    result = []
    src = cv2.VideoCapture(video_path)
    video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Random start point for frame selection
    start = random.randint(0, max(0, video_length - 1 - (n_frames - 1) * frame_step))
    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    
    ret, frame = src.read()
    if ret:
        result.append(format_frames(frame, output_size))
        
        for _ in range(n_frames - 1):
            for _ in range(frame_step):
                src.read()  # Skip frames
            ret, frame = src.read()
            if ret:
                result.append(format_frames(frame, output_size))
            else:
                result.append(tf.zeros_like(result[0]))  # Add black frame if video ends

    src.release()
    return np.array(result)[..., [2, 1, 0]]  # Convert BGR to RGB

def load_and_process_data():
    file_paths, targets = [], []
    for i, cls in enumerate(CLASSES):
        video_files = glob.glob(f"{DATASET_PATH}/{cls}/**.avi")
        file_paths.extend(video_files)
        targets.extend([i] * len(video_files))
    
    features = np.array([frames_from_video_file(f, n_frames=10) for f in file_paths])
    targets = np.array(targets)
    return features, targets