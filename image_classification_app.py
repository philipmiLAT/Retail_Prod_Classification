import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

#Trained Model (TO DO)
#MODEL_PATH = "" #location of trained model file
#IMAGE_SIZE = () #input size of image the model was trained on (width, height)
#CLASS_NAMES = [] #categories our model can predict. Make sure they are in the same order as when training the model


#Loading the Trained Model (TO DO)
#st.cache_resource
#def load_model():
    #comment:loads model only when needed
    #from tensorflow.keras.models import load_model as keras_load
    #return keras_load(MODEL_PATH)


#Preprocess and Predict (TO DO)
#This turns the picture into the exact format the model expects
#def preprocess(img: Image.Image)->np.ndarray:
 #   img = img.convert("RGB".resize(IMAGE_SIZE))
 #   arr = np.array(img, dtype="")  #converts image into a grid of numbers. Pixels are 0-255 (must match how model was trained)
  #  return

#This runs an image through the model and returns the label and confidence
#def classify(model, img: Image.Image):
  #  probs = model.predit(preprocess(img), verbose=0)[0]
  #  idx = int(np.argmax(probs))
   # return CLASS_NAMES[idx], float(probs[idx]) #returns category and the confidence




#APP Interface
st.set_page_config(page_title="Product Classifier", layout="wide")

st.title("Product Classifier")
st.write("This app makes it easier to categorize inventory, saving you valuable time!")

st.write("It's simple! Just upload product photos and this app will categorize each one for you.")

#This helps flags uncertain predictions
with st.sidebar: 
    st.header("Settings")
    threshold = st.slider("Flag prediction below this confidence", 0.0, 1.0, 0.6)

#Upload product images
uploaded_images = st.file_uploader("Upload your product images:", 
                                   type=["jpg","jpeg", "png"], 
                                   accept_multiple_files=True)

#runs when at least one image has been uploaded
if uploaded_images:
    #displays green banned with the count of images
    st.success(f"Loaded {len(uploaded_images)} images!")

    #Returns true when buttonw is clicked and classification starts
    if st.button("Classify my products"):
        model = load_model() #loads the model (need to finish code at the top)
        results = [] #per image

        #Optional: progress bar of classification
        progress = st.progress(0, text="Classifying...")

        #loops over every uploaded image
        for i, file in enumerate(uploaded_images):
            img = Image.open(file) #opens the image from file
            label, conf = classify(model, img) #classify the image (need to finish code above to upload model)

            #saves everything so we can display results later
            results.append({"File": file.name, 
                            "Category": label, 
                            "Confidence": conf, 
                            "Image":img})
            
            #this moves the progress bar forward
            progress.progress((i+1)/len(uploaded_images), text=f"Classified {i + 1} of {len(uploaded_images)}")

        progress.empty() #removes the progress bar when finished



#Summary section of classifies items

#this makes a table from the results. Table of file names, categories, and confidence.
df = pd.DataFrame(results.drop(columns="Image"))

#3 summary numbers
c1, c2, c3 = st.columns(3)
c1.metric("Products classified", len(df))
c2.metric("Categories found", df["Category"].nunique()) #unique labels
c3.metric("Low confidence", int((df["Confidence"] < threshold).sum())) #counts how many predicitons fall below the slider value

#Bar chart: value_counts() counts how many products are in each category
st.bar_chart(df["Category"].value_counts())


#Image Grid: displays each prodcut with its predicted category
st.subheader("Results")
cols = st.columns(4)
for i, r in enumerate(results):
    with cols[i % 4]: #images are put into grid
        st.image(r["Image"], use_container_width=True)
        st.markdown(f"**{r['Category']}**") #bolds category name
        msg = f"{r['Confidence']:.0%} confident" #displays confidence
        if r["Confidence"] < threshold:
            st.warning(msg) #displays yellow box for uncertain predictions
        else:
            st.caption(msg) #displays small gray text for confident predictions


#Table view and download
st.subheader("Table view")
st.dataframe(df, use_container_width=True) #interactive, sortable table

st.download_button(
    "Download results as CSV", 
    df.to_csv(index=False), #table converted to CSV text
    "classified_products.csv", #file name user will get
    "text/csv", #file type
)


