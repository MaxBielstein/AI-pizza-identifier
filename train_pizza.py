#!/usr/bin/env python3
import argparse, json, numpy as np, tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

SEED = 42
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Cheese", "Pepperoni", "Hamburger"]

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "pizza_dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def infer_label_from_filename(image_path):
    mapping = {"c": "Cheese", "p": "Pepperoni", "h": "Hamburger"}
    return mapping.get(image_path.name[:1].lower())

def load_labeled_items():
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    labeled_items = []
    for path in sorted(DATASET_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            label_name = infer_label_from_filename(path)
            if label_name:
                labeled_items.append((path, CLASS_NAMES.index(label_name)))
    if not labeled_items:
        raise SystemExit("No labeled images in pizza_dataset.")
    return labeled_items

def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    return {i: (total / (num_classes * counts[i])) for i in range(num_classes)}

def build_dataset(pairs, batch_size, training, class_weight_vector=None):
    image_paths = tf.constant([str(p) for p, _ in pairs])
    label_indices = tf.constant([y for _, y in pairs], dtype=tf.int32)

    def _load(image_path, label_index):
        image = tf.io.decode_image(tf.io.read_file(image_path), channels=3, expand_animations=False)
        image = tf.image.resize_with_pad(image, *IMAGE_SIZE)
        image = preprocess_input(tf.cast(image, tf.float32))
        weight = 1.0 if class_weight_vector is None else tf.cast(class_weight_vector[label_index], tf.float32)
        return image, label_index, weight

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_indices)).map(
        _load, num_parallel_calls=tf.data.AUTOTUNE
    )
    if training:
        augmenter = tf.keras.Sequential([layers.RandomFlip("horizontal")])
        dataset = dataset.map(
            lambda img, lbl, w: (augmenter(img, training=True), lbl, w),
            num_parallel_calls=tf.data.AUTOTUNE
        ).shuffle(len(pairs), seed=SEED, reshuffle_each_iteration=True)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def build_model(num_classes):
    backbone = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", pooling="avg")
    backbone.trainable = False
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    features = backbone(inputs, training=False)
    logits = layers.Dense(num_classes, activation="softmax")(features)
    model = models.Model(inputs, logits)
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def print_dataset_split(split_name, pairs):
    by_class = {c: [] for c in CLASS_NAMES}
    for path, label_idx in pairs:
        by_class[CLASS_NAMES[label_idx]].append(path.name)
    print(f"\n{split_name} ({len(pairs)} images)")
    for class_name in CLASS_NAMES:
        files = by_class[class_name]
        print(f"  {class_name}: {len(files)}")
        for filename in files:
            print(f"    - {filename}")

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--epochs", type=int, default=20)
    args = arg_parser.parse_args()

    rng = np.random.default_rng(SEED)
    labeled_items = load_labeled_items()

    indices = np.arange(len(labeled_items))
    rng.shuffle(indices)

    split_index = max(1, int(0.8 * len(labeled_items)))
    train_samples = [labeled_items[i] for i in indices[:split_index]]
    val_samples   = [labeled_items[i] for i in indices[split_index:]]

    print_dataset_split("TRAIN", train_samples)
    print_dataset_split("VAL",   val_samples)

    train_label_indices = np.array([y for _, y in train_samples])
    weight_map = compute_class_weights(train_label_indices, len(CLASS_NAMES))
    class_weight_vector = tf.constant([weight_map[i] for i in range(len(CLASS_NAMES))], dtype=tf.float32)

    train_ds = build_dataset(train_samples, batch_size=4, training=True,  class_weight_vector=class_weight_vector)
    val_ds   = build_dataset(val_samples,   batch_size=4, training=False, class_weight_vector=None)

    (OUTPUT_DIR / "class_index.json").write_text(json.dumps({c: i for i, c in enumerate(CLASS_NAMES)}))

    model = build_model(len(CLASS_NAMES))
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=1)
    print(model.evaluate(val_ds, return_dict=True))
    model.save(OUTPUT_DIR / "final_model.keras")

if __name__ == "__main__":
    main()
