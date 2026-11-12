import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# ---------------------------------------------------------------
# Trained model settings
# Keep app.py, mobilenetv2_retail.keras and class_names.json in the same folder.
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR.parent / "models" / "mobilenetv2_retail.keras"
CLASS_NAMES_PATH = BASE_DIR.parent / "models" / "class_names.json"    # category order used in training
IMAGE_SIZE = (224, 224)                              # (width, height) the model was trained on


@st.cache_resource
def load_class_names():
    """Return a list where list[i] is the category name for model output index i."""
    with open(CLASS_NAMES_PATH) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if all(str(k).isdigit() for k in data.keys()):  # {"0": "shoes", ...}
        return [data[str(i)] for i in range(len(data))]
    names = [None] * len(data)                        # {"shoes": 0, ...}
    for name, idx in data.items():
        names[int(idx)] = name
    return names


@st.cache_resource
def load_model():
    # Loads the model once, then reuses it on every rerun
    from tensorflow.keras.models import load_model as keras_load
    return keras_load(MODEL_PATH)


def preprocess(img: Image.Image) -> np.ndarray:
    """Turn a picture into the exact format the model expects."""
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype="float32")  # pixels stay 0-255; the model rescales them itself
    return np.expand_dims(arr, 0)         # shape (1, 224, 224, 3): a batch of one image


def classify(model, img: Image.Image):
    """Run one image through the model; return (category, confidence)."""
    probs = model.predict(preprocess(img), verbose=0)[0]
    idx = int(np.argmax(probs))
    return load_class_names()[idx], float(probs[idx])


# ---------------------------------------------------------------
# App interface
# ---------------------------------------------------------------
st.set_page_config(page_title="Product Classifier", layout="wide")

st.title("Product Classifier")
st.write("This app makes it easier to categorize inventory, saving you valuable time!")
st.write("It's simple! Just upload product photos and this app will categorize each one for you.")

# Flags uncertain predictions
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Flag prediction below this confidence", 0.0, 1.0, 0.6)

uploaded_images = st.file_uploader(
    "Upload your product images:",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_images:
    st.success(f"Loaded {len(uploaded_images)} images!")

    if st.button("Classify my products"):
        model = load_model()
        results = []

        progress = st.progress(0, text="Classifying...")
        for i, file in enumerate(uploaded_images):
            img = Image.open(file)
            label, conf = classify(model, img)
            results.append({"File": file.name, "Category": label,
                            "Confidence": conf, "Image": img})
            progress.progress((i + 1) / len(uploaded_images),
                              text=f"Classified {i + 1} of {len(uploaded_images)}")
        progress.empty()

        # Streamlit reruns the whole script when you move the slider,
        # so keep the results in session_state or they would disappear.
        st.session_state["results"] = results

# ---------------------------------------------------------------
# Results (only shown once something has been classified)
# ---------------------------------------------------------------
results = st.session_state.get("results")

if results:
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "Image"} for r in results])

    c1, c2, c3 = st.columns(3)
    c1.metric("Products classified", len(df))
    c2.metric("Categories found", df["Category"].nunique())
    c3.metric("Low confidence", int((df["Confidence"] < threshold).sum()))

    st.bar_chart(df["Category"].value_counts())

    st.subheader("Results")
    cols = st.columns(4)
    for i, r in enumerate(results):
        with cols[i % 4]:
            st.image(r["Image"])
            st.markdown(f"**{r['Category']}**")
            msg = f"{r['Confidence']:.0%} confident"
            if r["Confidence"] < threshold:
                st.warning(msg)
            else:
                st.caption(msg)

    st.subheader("Table view")
    st.dataframe(df)

    st.download_button(
        "Download results as CSV",
        df.to_csv(index=False),
        "classified_products.csv",
        "text/csv",
    )
