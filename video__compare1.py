import cv2
import numpy as np
from PIL import Image
import imagehash
from concurrent.futures import ThreadPoolExecutor
import os

def is_dark_frame(img, threshold=10):
    return np.mean(img) < threshold
def extract_frames(video_path, output_dir, fps=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not is_dark_frame(gray_frame):
                timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)  # Calculate timestamp in seconds
                frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}_{timestamp:.3f}.jpg")
                cv2.imwrite(frame_path, frame)
        
        frame_count += 1
    
    cap.release()

def is_solid_frame(image_path, threshold=10):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    return np.std(img) < threshold

def compute_phash(image_path):
    try:
        return str(imagehash.phash(Image.open(image_path)))
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None




def compare_videos(original_dir, infringing_dir, hash_difference_threshold=13):
    original_frames = [f for f in os.listdir(original_dir) if f.endswith('.jpg') and not is_solid_frame(os.path.join(original_dir, f))]
    infringing_frames = [f for f in os.listdir(infringing_dir) if f.endswith('.jpg')]

    with ThreadPoolExecutor() as executor:
        original_hashes = list(executor.map(lambda f: (f, compute_phash(os.path.join(original_dir, f))), original_frames))
        infringing_hashes = list(executor.map(lambda f: (f, compute_phash(os.path.join(infringing_dir, f))), infringing_frames))

    matches = []
    for orig_frame, orig_hash in original_hashes:
        if orig_hash is None:
            continue
        for infr_frame, infr_hash in infringing_hashes:
            if infr_hash is None:
                continue
            hash_difference = imagehash.hex_to_hash(orig_hash) - imagehash.hex_to_hash(infr_hash)
            if hash_difference < hash_difference_threshold:
                similarity = 1 - (hash_difference / 64)  # Convert hash difference to similarity score
                matches.append((orig_frame, infr_frame, similarity))

    return matches
# Main execution
original_video = "./output.mp4"
infringing_video = "./TP_TOUR_030495_Lou_Hr9.mp4"
original_frames_dir = "./original_frames"
infringing_frames_dir = "./infringed_frames1"

# Extract frames
#extract_frames(original_video, original_frames_dir)
extract_frames(infringing_video, infringing_frames_dir)

# Compare videos
matches = compare_videos(original_frames_dir, infringing_frames_dir)

# Print results
for orig_frame, infr_frame, similarity in sorted(matches, key=lambda x: x[2], reverse=True):
    # Extract frame number and timestamp from filenames
    orig_frame_num = int(orig_frame.split('_')[1])
    orig_timestamp = float(orig_frame.split('_')[2].split('.')[0])
    infr_frame_num = int(infr_frame.split('_')[1])
    infr_timestamp = float(infr_frame.split('_')[2].split('.')[0])
    
    # Convert timestamps to minutes and seconds
    orig_minutes, orig_seconds = divmod(orig_timestamp, 60)
    infr_minutes, infr_seconds = divmod(infr_timestamp, 60)
    
    print(f"Match found: Original video frame {orig_frame_num} at {orig_minutes:.0f}m {orig_seconds:.2f}s matches "
          f"Infringing video frame {infr_frame_num} at {infr_minutes:.0f}m {infr_seconds:.2f}s (Similarity: {similarity:.4f})")