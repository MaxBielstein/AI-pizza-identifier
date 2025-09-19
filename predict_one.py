#!/usr/bin/env python3
import argparse, json, numpy as np, tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMAGE_SIZE = (224, 224)

def load_and_preprocess_image(image_path):
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize_with_pad(image, *IMAGE_SIZE)
    image = preprocess_input(tf.cast(image, tf.float32))
    return tf.expand_dims(image, 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent / "outputs" / "final_model.keras")
    )
    parser.add_argument(
        "--class_map",
        default=str(Path(__file__).resolve().parent / "outputs" / "class_index.json")
    )
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)

    class_index_map = json.loads(Path(args.class_map).read_text())
    inverse_class_map = {int(v): k for k, v in class_index_map.items()}
    class_names = [inverse_class_map[i] for i in range(len(inverse_class_map))]

    input_image = load_and_preprocess_image(args.image)
    probabilities = model.predict(input_image, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    print(f"Image: {args.image}")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:10s}: {probabilities[i]:.4f}")
    print(f"Predicted: {class_names[predicted_index]}")

if __name__ == "__main__":
    main()
