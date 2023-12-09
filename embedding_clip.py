#!/usr/bin/python3
# Description: This script generates CLIP embeddings for images or text.
# Usage:
# As an example, let’s convert the text "a sleepy ridgeback dog" into an embedding. 
# For purposes of brevity, we have cropped the full embedding result which can be found here.
# python embedding_clip.py --text "a sleepy ridgeback dog"
# python embedding_clip.py --image "./images/*.jpg" --output output.csv
# [0.5736801028251648, 0.2516217529773712, ...,  -0.6825592517852783]
import argparse
import csv
from PIL import Image
import clip
import torch
import time
import os
import re
import hashlib

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

def write_to_csv(file_path, input_data, time_taken, md5_hash, embedding):
    with open(file_path, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([input_data, time_taken, md5_hash, embedding])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generate',
        description='Generate CLIP embeddings for images or text')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', required=False)
    group.add_argument('--image', required=False)
    parser.add_argument('--output', default='clip.csv', help='Output CSV file path')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    device = torch.device(device)
    model, preprocess = clip.load("ViT-L/14")
    model.to(device)

    if args.text:
        start_time = time.time()
        print(f"Gathering embeddings for {args.text}")
        inputs = clip.tokenize(args.text)
        with torch.no_grad():
            embedding = model.encode_text(inputs)[0].tolist()
        time_taken = time.time() - start_time
        write_to_csv(args.output, args.text, time_taken, 'N/A', embedding)

    elif args.image:
        # Check if image_path is a file or a pattern
        if os.path.isfile(args.image):
            image_paths = [args.image]
        else:
            directory, pattern = os.path.split(args.image)
            if not directory:
                directory = '.'
            image_paths = get_matching_files(directory, pattern)

        # Check if the file already exists, if not create it
        if not os.path.isfile(args.output):
            with open(args.output, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['image', 'seconds', 'md5', 'embedding'])
            
        for idx, image_path in enumerate(image_paths):
            start_time = time.time()
            print(f"Processing image {idx + 1} of {len(image_paths)}: {image_path}")
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image)[0].tolist()
            time_taken = time.time() - start_time
            md5_hash = get_md5_hash(image_path)
            write_to_csv(args.output, image_path, time_taken, md5_hash, embedding)

    print("Processing complete.")
