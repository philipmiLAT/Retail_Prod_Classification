"""
Train a MobileNetV2 classifier on the retail product dataset.

Usage:
    python train_mobilenetv2.py --root /path/to/extracted/retail_dataset

Expects under --root:
    manifest.csv       (image id, category label, split, sha256, relative path)
    class_names.json   (defines the category -> index mapping)
    train/ val/ test/  (images)

The test split is NOT touched until the final evaluation step.
"""
import argparse
import json
import os

import pandas as pd
import tensorflow as tf

IMG_SIZE = (224, 224)


def find_col(df, candidates, what):
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    raise KeyError(f"Could not find a {what} column in manifest. Columns: {list(df.columns)}")


def load_class_names(path):
    """Return a list where list[i] is the class name for index i."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        keys_are_ints = all(str(k).isdigit() for k in data.keys())
        if keys_are_ints:  # {"0": "shoes", ...}
            return [data[str(i)] for i in range(len(data))]
        # {"shoes": 0, ...}
        out = [None] * len(data)
        for name, idx in data.items():
            out[int(idx)] = name
        return out
    raise ValueError("Unrecognized class_names.json format")


def make_dataset(df, root, path_col, label_col, class_to_idx, training, batch_size):
    paths = [os.path.join(root, p) for p in df[path_col]]
    labels = df[label_col].map(class_to_idx).astype("int32").values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)  # force RGB
        img = tf.image.resize(img, IMG_SIZE)
        img.set_shape((*IMG_SIZE, 3))
        return img, label

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)       # active only in training
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    # Required MobileNetV2 preprocessing: scales pixels from [0, 255] to [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--head-epochs", type=int, default=8)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--out", default="mobilenetv2_retail.keras")
    args = ap.parse_args()

    class_names = load_class_names(os.path.join(args.root, "class_names.json"))
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    df = pd.read_csv(os.path.join(args.root, "manifest.csv"))
    print("Manifest columns:", list(df.columns))
    path_col = find_col(df, ["relative_path", "path", "image_path", "filepath", "file_path"], "path")
    label_col = find_col(df, ["categories", "category", "label", "class", "category_label"], "category")
    split_col = find_col(df, ["split", "assigned_split"], "split")

    unknown = set(df[label_col]) - set(class_to_idx)
    if unknown:
        raise ValueError(f"Labels in manifest not in class_names.json: {sorted(unknown)[:10]}")

    splits = {s: df[df[split_col] == s] for s in ("train", "val", "test")}
    for s, d in splits.items():
        print(f"{s}: {len(d)} images")

    train_ds = make_dataset(splits["train"], args.root, path_col, label_col, class_to_idx, True, args.batch_size)
    val_ds = make_dataset(splits["val"], args.root, path_col, label_col, class_to_idx, False, args.batch_size)

    model, base = build_model(len(class_names))
    ckpt = tf.keras.callbacks.ModelCheckpoint(args.out, monitor="val_accuracy", save_best_only=True)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)

    # Phase 1: train the new classification head
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs, callbacks=[ckpt, early])

    # Phase 2: fine-tune the top of the backbone at a low learning rate
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs, callbacks=[ckpt, early])

    # Final evaluation on the held-out test split (first and only use)
    test_ds = make_dataset(splits["test"], args.root, path_col, label_col, class_to_idx, False, args.batch_size)
    loss, acc = model.evaluate(test_ds)
    print(f"TEST loss={loss:.4f} accuracy={acc:.4f}")


if __name__ == "__main__":
    main()