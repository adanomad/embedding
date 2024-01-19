#! /usr/bin/env python3

# This script is a Flask app that exposes two endpoints:
# 1. /embed: Takes a text input and returns the CLIP embedding
# 2. /image: Takes a filename and returns the image

import glob
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import clip
from PIL import Image
import os
import io


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
device = torch.device(device)
model, preprocess = clip.load("ViT-L/14", device=device)

mount = os.getenv("MOUNT", "")

@app.route("/embed", methods=["POST"])
def get_embedding():
    data = request.json
    input_text = data.get("text")
    print(f"text: {input_text}")

    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    with torch.no_grad():
        inputs = clip.tokenize([input_text]).to(device)
        embedding = model.encode_text(inputs)
        embedding_list = embedding[0].cpu().numpy().tolist()

    return jsonify({"embedding": embedding_list})


@app.route("/image", methods=["GET"])
def get_image():
    filename = request.args.get("fileName")

    if not filename:
        return jsonify({"error": "No fileName provided"}), 400

    # Extract date from filename and construct the path
    date_folder = filename[:8]  # Assuming date is in 'YYYYMMDD' format
    image_path = f"{mount}/ss/{date_folder}/{filename}"

    if not os.path.exists(image_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(image_path, mimetype="image/jpeg")


@app.route("/list_files_by_date", methods=["GET"])
def list_files_by_date():
    filename = request.args.get("fileName")

    if not filename or len(filename) < 8:
        return jsonify({"error": "Invalid or no fileName provided"}), 400

    # Extract date from filename and construct the path
    date_folder = filename[:8]  # Assuming date is in 'YYYYMMDD' format
    directory_path = f"{mount}/ss/{date_folder}/"

    if not os.path.exists(directory_path):
        return jsonify({"error": "Directory not found"}), 404

    # List all files in the directory
    files = [f for f in glob.glob(directory_path + "*") if os.path.isfile(f)]
    file_names = [os.path.basename(f) for f in files]

    return jsonify({"files": file_names})


def create_thumbnail(image_path, max_size=(600, 600)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        byte_arr = io.BytesIO()
        img.save(byte_arr, format="JPEG")
        byte_arr = byte_arr.getvalue()
    return byte_arr


# users will initially view thumbnails and can opt to see the full-size image on demand.
# Generating thumbnails on the fly for each request might not be efficient for a production server with high traffic.
# Ideally generating and caching thumbnails ahead of time.
@app.route("/thumbnail", methods=["GET"])
def get_thumbnail():
    filename = request.args.get("fileName")

    if not filename:
        return jsonify({"error": "No fileName provided"}), 400

    date_folder = filename[:8]
    image_path = f"{mount}/{date_folder}/{filename}"

    if not os.path.exists(image_path):
        return jsonify({"error": "File not found"}), 404

    thumbnail = create_thumbnail(image_path)
    return send_file(io.BytesIO(thumbnail), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
