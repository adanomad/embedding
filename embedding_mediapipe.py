#!/usr/bin/python3
# Description: This script generates an embedding for image.
# The downside is there's not text embedding, so you can't compare images to text.
# But the upside is that it is lighter-weight, faster than CLIP.
# mobilenet_v3_small.tflite is 3.9M
# mobilenet_v3_large.tflite is 10M
# Usage:
# python embedding_mediapipe.py mobilenet_v3_small.tflite image.jpg

import mediapipe as mp
import time
import argparse
import re
import os
import csv
import hashlib
from typing import Tuple

# Typing for MediaPipe classes
BaseOptions = mp.tasks.BaseOptions
ImageEmbedder = mp.tasks.vision.ImageEmbedder
ImageEmbedderOptions = mp.tasks.vision.ImageEmbedderOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def embed_image(model_path: str, mp_image_path: str) -> Tuple[list, float]:
    """
    Embeds an image using the specified model and returns the embedding vector and time taken.

    :param model_path: Path to the model file.
    :param mp_image_path: Path to the image file.
    :return: Tuple of time taken and the embedding vector.
    """
    options = ImageEmbedderOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        quantize=True,
        running_mode=VisionRunningMode.IMAGE)

    start_time = time.time()

    with ImageEmbedder.create_from_options(options) as embedder:
        mp_image = mp.Image.create_from_file(mp_image_path)
        embedding_result = embedder.embed(mp_image)
        vector = embedding_result.embeddings[0].embedding

    end_time = time.time()
    return list(vector), end_time - start_time

def get_matching_files(directory, pattern):
    """
    Returns a list of file paths in 'directory' that match the 'pattern'.
    """
    matching_files = []
    for f in os.listdir(directory):
        if re.match(pattern, f):
            matching_files.append(os.path.join(directory, f))
    return matching_files



def get_md5_hash(file_name: str) -> str:
    """Generate MD5 hash for a given file."""
    with open(file_name, 'rb') as file:
        return hashlib.md5(file.read()).hexdigest()



def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for images using MediaPipe.")
    parser.add_argument('--model_path', type=str, help='Path to the model file.')
    parser.add_argument('--image_path', type=str, help='Path to the image file or a regex pattern for files.')
    parser.add_argument('--output_csv', type=str, help='Path to the output CSV file.')
    args = parser.parse_args()

    model_path = args.model_path

    # Check if image_path is a file or a pattern
    if os.path.isfile(args.image_path):
        # It's a single file
        image_paths = [args.image_path]
    else:
        # Extract the directory and the pattern
        directory, pattern = os.path.split(args.image_path)

        # If the directory is empty, assume the current directory
        if not directory:
            directory = '.'

        # Get the list of matching file paths
        image_paths = get_matching_files(directory, pattern)

    with open(args.output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'seconds', 'md5', 'embedding'])

        for idx, mp_image_path in enumerate(image_paths):
            print(f'Processing image {idx + 1} of {len(image_paths)}: {mp_image_path}')
            vector, time_taken = embed_image(model_path, mp_image_path)
            hash = get_md5_hash(mp_image_path)
            writer.writerow([mp_image_path, time_taken, hash, vector])


if __name__ == "__main__":
    main()
