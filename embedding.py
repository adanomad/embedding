import mediapipe as mp
import time
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
    return vector, end_time - start_time

def main():
    model_small_path = './mobilenet_v3_small.tflite'
    model_large_path = './mobilenet_v3_large.tflite'
    mp_image_path = './sample.jpg'

    # Benchmark small model
    vector_small, time_small = embed_image(model_small_path, mp_image_path)
    print(f"Small Model Time: {time_small} seconds")
    print(f"Small Model Embedding: {vector_small}")

    # Benchmark large model
    vector_large, time_large = embed_image(model_large_path, mp_image_path)
    print(f"Large Model Time: {time_large} seconds")
    print(f"Large Model Embedding: {vector_large}")

if __name__ == "__main__":
    main()
