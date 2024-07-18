import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
def preprocess_image(image_path, size=(128, 128)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    try:
        img = Image.open(image_path)
        return transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def compare_videos(original_dir, infringing_dir, similarity_threshold=0.8, batch_size=64):
    original_frames = sorted([f for f in os.listdir(original_dir) if f.endswith('.jpg')])
    infringing_frames = sorted([f for f in os.listdir(infringing_dir) if f.endswith('.jpg')])

    matches = []

    # Preprocess all images
    with ThreadPoolExecutor() as executor:
        orig_images = list(executor.map(lambda f: (f, preprocess_image(os.path.join(original_dir, f))), original_frames))
        infr_images = list(executor.map(lambda f: (f, preprocess_image(os.path.join(infringing_dir, f))), infringing_frames))

    # Remove None values
    orig_images = [(f, img) for f, img in orig_images if img is not None]
    infr_images = [(f, img) for f, img in infr_images if img is not None]

    # Batch processing
    for i in range(0, len(orig_images), batch_size):
        orig_batch = orig_images[i:i+batch_size]
        orig_tensors = torch.cat([img for _, img in orig_batch])

        for j in range(0, len(infr_images), batch_size):
            infr_batch = infr_images[j:j+batch_size]
            infr_tensors = torch.cat([img for _, img in infr_batch])

            # Compute similarities for the entire batch
            similarities = compute_similarity_batch(orig_tensors, infr_tensors)

            # Find matches
            matches_batch = (similarities > similarity_threshold).nonzero(as_tuple=True)
            for orig_idx, infr_idx in zip(matches_batch[0], matches_batch[1]):
                orig_file = orig_batch[orig_idx.item()][0]
                infr_file = infr_batch[infr_idx.item()][0]
                similarity = similarities[orig_idx, infr_idx].item()
                matches.append((orig_file, infr_file, similarity))

    return matches

def compute_similarity_batch(batch1, batch2):
    # Structural Similarity (using MSE as a simple approximation)
    mse = torch.cdist(batch1.view(batch1.size(0), -1), batch2.view(batch2.size(0), -1))
    ssim_score = 1 - mse / mse.max()  # Normalize and invert

    # Perceptual Hash (using L2 distance as an approximation)
    l2_distance = torch.cdist(batch1.view(batch1.size(0), -1), batch2.view(batch2.size(0), -1))
    phash_score = 1 / (1 + l2_distance)  # Higher score means more similar

    return 0.5 * ssim_score + 0.5 * phash_score


# Main execution
original_video = "./output.mp4"
infringing_video = "./TP__II_111795_R1.mp4"
original_frames_dir = "./original_frames"
infringing_frames_dir = "./infringed_frames"

# Extract frames (this part remains CPU-bound)
extract_frames(original_video, original_frames_dir, fps=1)
extract_frames(infringing_video, infringing_frames_dir, fps=1)

# Compare videos
matches = compare_videos(original_frames_dir, infringing_frames_dir, similarity_threshold=0.83)

# Print results
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