import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
ImageEmbedder = mp.tasks.vision.ImageEmbedder
ImageEmbedderOptions = mp.tasks.vision.ImageEmbedderOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_small_path = './mobilenet_v3_small.tflite'
model_large_path = './mobilenet_v3_large.tflite'
model_path = model_small_path

options = ImageEmbedderOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    quantize=True,
    running_mode=VisionRunningMode.IMAGE)

    
if __name__ == "__main__":
    # Load the input image from an image file.
    mp_image = './sample.jpg'

    with ImageEmbedder.create_from_options(options) as embedder:
        # The embedder is initialized. Use it here.
        mp_image = mp.Image.create_from_file(mp_image)
        # Perform image embedding on the provided single image.
        embedding_result = embedder.embed(mp_image)
        vector = embedding_result.embeddings[0].embedding
        print("Embedding:", vector)
