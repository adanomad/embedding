#! /usr/bin/env python3

# This script is a Flask app that exposes two endpoints:
# 1. /embed: Takes a text input and returns the CLIP embedding
# 2. /image: Takes a filename and returns the image

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import clip
from PIL import Image
import time
import os


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
device = torch.device(device)
model, preprocess = clip.load("ViT-L/14", device=device)


@app.route('/embed', methods=['POST'])
def get_embedding():
    data = request.json
    input_text = data.get("text")
    print(f"text: {input_text}")

    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    with torch.no_grad():
        inputs = clip.tokenize([input_text]).to(device)
        embedding = model.encode_text(inputs)
        embedding_list = embedding[0].cpu().numpy().tolist()  # Convert to list

    return jsonify({"embedding": embedding_list})


@app.route('/image', methods=['GET'])
def get_image():
    filename = request.args.get('fileName')

    if not filename:
        return jsonify({"error": "No fileName provided"}), 400

    # Extract date from filename and construct the path
    date_folder = filename[:8]  # Assuming date is in 'YYYYMMDD' format
    image_path = f"/mnt/datahaus-jason/ss/{date_folder}/{filename}"

    if not os.path.exists(image_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(image_path, mimetype='image/jpeg')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
