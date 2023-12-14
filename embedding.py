#!/usr/bin/python3
# Description: Calculate CLIP embeddings for images or text, or MediaPipe embeddings for images.
# Usage:
# python embedding.py --text "a sleepy ridgeback dog"
# python embedding.py --image "./images/*.jpg" --output output.csv
# python embedding.py --image "path/to/image.jpg" --output output.csv --embedding_method mediapipe

# CLIP embedder runs faster (10ms) than the MediaPipe embedder (50ms) when CUDA is available.
# When CUDA is not available, CLIP embedder runs slower (3s).

import argparse
import csv
from PIL import Image
import clip
import torch
import time
import os
import re
import hashlib
import mediapipe as mp
from typing import Tuple
import glob


def get_md5_hash_and_size(file_name: str) -> [str, int]:
    """Generate MD5 hash for a given file and return its size."""
    with open(file_name, "rb") as file:
        file_content = file.read()
        file_size = os.path.getsize(file_name)
        md5_hash = hashlib.md5(file_content).hexdigest()
        return md5_hash, file_size



def write_to_csv(file_path, input_data, time_taken, file_size, md5_hash, embedding):
    with open(file_path, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([input_data, time_taken, file_size, md5_hash, embedding])


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
    def __init__(self, model_path: str):
        options = mp.tasks.vision.ImageEmbedderOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            quantize=True,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self.embedder = mp.tasks.vision.ImageEmbedder.create_from_options(options)

    def embed_image(self, mp_image_path: str) -> Tuple[list, float]:
        """
        Embeds an image using the specified model and returns the embedding vector and time taken.

        :param model_path: Path to the model file.
        :param mp_image_path: Path to the image file.
        :return: Tuple of time taken and the embedding vector.
        """
        start_time = time.time()
        mp_image = mp.Image.create_from_file(mp_image_path)
        embedding_result = self.embedder.embed(mp_image)
        vector = embedding_result.embeddings[0].embedding
        end_time = time.time()
        return list(vector), end_time - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate", description="Generate embeddings for images or text"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", required=False)
    group.add_argument("--image", required=False)
    parser.add_argument("--output", default="clip.csv", help="Output CSV file path")
    parser.add_argument(
        "--embedding_method",
        default="clip",
        choices=["clip", "mediapipe"],
        help="Choose embedding method: clip or mediapipe",
    )
    parser.add_argument(
        "--model_path",
        default="mobilenet_v3_large.tflite",
        help="Path to the model file for MediaPipe",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    device = torch.device(device)
    model, preprocess = clip.load("ViT-L/14")
    model.to(device)

    begin_time = time.time()
    if args.text:
        embedding, time_taken = embedding_clip(
            model, preprocess, device, args.text, is_text=True
        )
        write_to_csv(args.output, args.text, time_taken, 0, "N/A", embedding)

    elif args.image:
        if os.path.isfile(args.image):
            image_paths = [args.image]
        else:
            # Retrieve all files matching the pattern 
            image_paths = [
                file
                for file in glob.glob(args.image)
            ]
            image_paths.sort()  # Sort the files


        if not os.path.isfile(args.output):
            with open(args.output, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["image", "seconds", "size", "md5", "embedding"])

        mp_embedder = MediaPipeEmbedder(args.model_path)
        for idx, image_path in enumerate(image_paths):
            if args.embedding_method == "clip":
                embedding, time_taken = embedding_clip(
                    model, preprocess, device, image_path
                )
            elif args.embedding_method == "mediapipe":
                embedding, time_taken = mp_embedder.embed_image(image_path)

            md5_hash, file_size = get_md5_hash_and_size(image_path)
            write_to_csv(args.output, image_path, time_taken, file_size, md5_hash, embedding)
            print(
                f"Processed {idx + 1} of {len(image_paths)} in {time_taken:.2f}s: {image_path}"
            )

    print(f"Processing complete in {(time.time() - begin_time):.2f}s")
