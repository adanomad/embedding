#!/usr/bin/python3
# Description: Calculate CLIP embeddings for images or text, or MediaPipe embeddings for images.
# Usage:
# python embedding.py --text "a sleepy ridgeback dog"
# python embedding.py --image "./images/*.jpg" --output output.csv
# python embedding.py --image "path/to/image.jpg" --output output.csv --embedding_method mediapipe


import argparse
import csv
from functools import cache
from PIL import Image
import clip
import torch
import time
import os
import re
import hashlib
import mediapipe as mp
from typing import Tuple


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
    with open(file_name, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()


def write_to_csv(file_path, input_data, time_taken, md5_hash, embedding):
    with open(file_path, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([input_data, time_taken, md5_hash, embedding])


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


def embedding_mediapipe(model_path: str, mp_image_path: str) -> Tuple[list, float]:
    """
    Embeds an image using the specified model and returns the embedding vector and time taken.

    :param model_path: Path to the model file.
    :param mp_image_path: Path to the image file.
    :return: Tuple of time taken and the embedding vector.
    """

    # Typing for MediaPipe classes
    BaseOptions = mp.tasks.BaseOptions
    ImageEmbedder = mp.tasks.vision.ImageEmbedder
    ImageEmbedderOptions = mp.tasks.vision.ImageEmbedderOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ImageEmbedderOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        quantize=True,
        running_mode=VisionRunningMode.IMAGE,
    )

    start_time = time.time()

    with ImageEmbedder.create_from_options(options) as embedder:
        mp_image = mp.Image.create_from_file(mp_image_path)
        embedding_result = embedder.embed(mp_image)
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
        write_to_csv(args.output, args.text, time_taken, "N/A", embedding)

    elif args.image:
        if os.path.isfile(args.image):
            image_paths = [args.image]
        else:
            directory, pattern = os.path.split(args.image)
            if not directory:
                directory = "."
            image_paths = get_matching_files(directory, pattern)

        if not os.path.isfile(args.output):
            with open(args.output, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["image", "seconds", "md5", "embedding"])

        for idx, image_path in enumerate(image_paths):
            if args.embedding_method == "clip":
                embedding, time_taken = embedding_clip(
                    model, preprocess, device, image_path
                )
            elif args.embedding_method == "mediapipe":
                embedding, time_taken = embedding_mediapipe(args.model_path, image_path)

            md5_hash = get_md5_hash(image_path)
            write_to_csv(args.output, image_path, time_taken, md5_hash, embedding)
            print(
                f"Processed {idx + 1} of {len(image_paths)} in {time_taken:.2f}s: {image_path}"
            )

    print(f"Processing complete in {(time.time() - begin_time):.2f}s")
