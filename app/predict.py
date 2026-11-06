"""
Run the trained retail classifier on your own images.

Usage (PowerShell, from the folder with the model and class_names.json):
    python predict.py photo1.jpg photo2.png
    python predict.py "C:\\path\\to\\folder_of_images"
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


def load_class_names(path):
    """Return a list where list[i] is the class name for index i."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if all(str(k).isdigit() for k in data.keys()):  # {"0": "shoes", ...}
        return [data[str(i)] for i in range(len(data))]
    out = [None] * len(data)                          # {"shoes": 0, ...}
    for name, idx in data.items():
        out[int(idx)] = name
    return out


def load_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)  # force RGB
    img = tf.image.resize(img, IMG_SIZE)
    return img  # 0-255 floats; the model applies MobileNetV2 preprocessing itself


def collect_paths(inputs):
    paths = []
    for p in inputs:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.lower().endswith(EXTS):
                    paths.append(os.path.join(p, name))
        else:
            paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="image files and/or folders of images")
    ap.add_argument("--model", default="models\mobilenetv2_retail.keras")
    ap.add_argument("--class-names", default="data\class_names.json")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    class_names = load_class_names(args.class_names)

    for path in collect_paths(args.images):
        try:
            x = tf.expand_dims(load_image(path), 0)
        except Exception as e:
            print(f"\n{path}\n  could not read image: {e}")
            continue
        probs = model.predict(x, verbose=0)[0]
        top = np.argsort(probs)[::-1][: args.top]
        print(f"\n{path}")
        for i in top:
            print(f"  {class_names[i]:<30} {probs[i] * 100:5.1f}%")


if __name__ == "__main__":
    main()