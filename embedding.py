#!/usr/bin/python3
# Description: Calculate CLIP embeddings for images or text, or MediaPipe embeddings for images.
# Usage:
# python embedding.py --text "a sleepy ridgeback dog"
# python embedding.py --image "./images/*.jpg" --output output.csv
# python embedding.py --image "path/to/image.jpg" --output output.csv --method mediapipe

# CLIP embedder runs faster (10ms) than the MediaPipe embedder (50ms) when CUDA is available.
# When CUDA is not available, CLIP embedder runs slower (3s).

import argparse
from PIL import Image
import clip
import torch
import time
import os
import hashlib
import mediapipe as mp
from typing import Tuple
import glob
import sqlite3
import numpy as np
import shutil


def is_size_similar(size1, size2, threshold=0.02):
    return abs(size1 - size2) / ((size1 + size2) / 2) < threshold

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def create_database(db_name):
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS embeddings
                   (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    filename TEXT, 
                    seconds REAL, 
                    size INTEGER, 
                    md5 TEXT, 
                    method TEXT, 
                    embedding TEXT)"""
    )
    conn.commit()
    return conn


def insert_into_db(conn, filename, seconds, size, md5, method, embedding):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO embeddings (filename, seconds, size, md5, method, embedding) VALUES (?, ?, ?, ?, ?, ?)",
        (filename, seconds, size, md5, method, str(embedding)),
    )
    conn.commit()


def get_file_info(file_name: str) -> [str, int]:
    """Returns MD5 hash and size."""
    with open(file_name, "rb") as file:
        file_content = file.read()
        file_size = os.path.getsize(file_name)
        md5_hash = hashlib.md5(file_content).hexdigest()
        return md5_hash, file_size


def embedding_clip(model, preprocess, device, input_data, is_text=True):
    start_time = time.time()
    with torch.no_grad():
        if is_text:
            inputs = clip.tokenize(input_data).to(
                device
            )  # Move tokenized inputs to the same device as the model
            embedding = model.encode_text(inputs)[0].tolist()
        else:
            image = preprocess(Image.open(input_data)).unsqueeze(0).to(device)
            embedding = model.encode_image(image)[0].tolist()
    time_taken = time.time() - start_time
    return embedding, time_taken


class MediaPipeEmbedder:
    def __init__(self, mediapipe_model_path: str):
        options = mp.tasks.vision.ImageEmbedderOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=mediapipe_model_path),
            quantize=True, # Quantize the model to reduce memory usage, but this can be funky
            # l2_normalize=True, # Normalize the embedding vector
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self.embedder = mp.tasks.vision.ImageEmbedder.create_from_options(options)

    def embed_image(self, mp_image_path: str) -> Tuple[list, float]:
        """
        Embeds an image using the specified model and returns the embedding vector and time taken.

        :param mediapipe_model_path: Path to the model file.
        :param mp_image_path: Path to the image file.
        :return: Tuple of time taken and the embedding vector.
        """
        start_time = time.time()
        mp_image = mp.Image.create_from_file(mp_image_path)
        embedding_result = self.embedder.embed(mp_image)
        vector = embedding_result.embeddings[0].embedding
        end_time = time.time()
        return vector, end_time - start_time


def setup_parser():
    parser = argparse.ArgumentParser(
        prog="generate", description="Generate embeddings for images or text"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", required=False)
    group.add_argument("--image", required=False)
    parser.add_argument(
        "--cosine_threshold",
        type=float,
        default=1.00,
        help="Threshold for cosine similarity",
    )
    parser.add_argument("--database", default="embedding.sqlite", help="Database name")
    parser.add_argument(
        "--method",
        choices=["clip", "mediapipe"],
        help="Choose embedding method: clip or mediapipe",
        required=True,
    )
    parser.add_argument(
        "--mediapipe_model_path",
        default="mobilenet_v3_large.tflite",
        help="Path to the model file for MediaPipe",
    )
    return parser.parse_args()


def initialize_embedding_environment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    device = torch.device(device)
    model, preprocess = clip.load("ViT-L/14")
    model.to(device)
    return device, model, preprocess


def process_text_embedding(args, conn, model, preprocess, device):
    embedding, time_taken = embedding_clip(
        model, preprocess, device, args.text, is_text=True
    )
    # Don't bother inserting into the database for a text embedding



def process_image_paths(image_arg):
    # Check if the argument is a file
    if os.path.isfile(image_arg):
        return [image_arg]

    # Expand user directory symbol if present
    image_arg = os.path.expanduser(image_arg)
    
    # Use glob to find files matching the pattern
    return sorted([file for file in glob.glob(image_arg) if os.path.isfile(file)])


def move_files(similar_files, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for file_name in similar_files:
        shutil.move(file_name, os.path.join(target_dir, os.path.basename(file_name)))


def process_image_embedding(args, conn, model, preprocess, device, mp_embedder):
    image_paths = process_image_paths(args.image)
    print(f"Processing {len(image_paths)} images")
    last_file_size = 0
    last_embedding = []
    for idx, image_path in enumerate(image_paths):
        # Compute the embedding vector and time taken
        if args.method == "clip":
            embedding, time_taken = embedding_clip(
                model, preprocess, device, image_path
            )
        elif args.method == "mediapipe":
            embedding, time_taken = mp_embedder.embed_image(image_path)
        md5_hash, file_size = get_file_info(image_path)
        basefilename = os.path.basename(image_path)

        # Be smart before doing more work. If the file size is too different, 
        # skip the rest.
        # Check if the file size is similar to the previous file
        if is_size_similar(file_size, last_file_size):
            distance = cosine_similarity(np.array(embedding), np.array(last_embedding))
            if distance > args.cosine_threshold:
                move_files([image_path], os.path.join(os.path.dirname(image_path), 'similar'))

        else:
            insert_into_db(
                conn, basefilename, time_taken, file_size, md5_hash, args.method, embedding
            )
        
        last_file_size = file_size
        last_embedding = embedding

        print(
            f"Processed {idx + 1} of {len(image_paths)} in {time_taken:.2f}s: {image_path}"
        )


def main():
    args = setup_parser()
    device, model, preprocess = initialize_embedding_environment()
    conn = create_database(args.database)

    begin_time = time.time()

    if args.text:
        process_text_embedding(args, conn, model, preprocess, device)
    elif args.image:
        mp_embedder = MediaPipeEmbedder(args.mediapipe_model_path)
        process_image_embedding(args, conn, model, preprocess, device, mp_embedder)

    conn.close()
    print(f"Processing complete in {(time.time() - begin_time):.2f}s")


if __name__ == "__main__":
    main()
