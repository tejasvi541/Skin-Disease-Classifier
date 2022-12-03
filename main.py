import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from PIL import Image, ImageOps
import numpy as np
 
classDict = [('Acne and Rosacea Photos', 0), ('Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 1), ('Atopic Dermatitis Photos', 2), ('Bullous Disease Photos', 3), ('Cellulitis Impetigo and other Bacterial Infections', 4), ('Eczema Photos', 5), ('Exanthems and Drug Eruptions', 6), ('Hair Loss Photos Alopecia and other Hair Diseases', 7), ('Herpes HPV and other STDs Photos', 8), ('Light Diseases and Disorders of Pigmentation', 9), ('Lupus and other Connective Tissue diseases', 10), ('Melanoma Skin Cancer Nevi and Moles', 11), ('Nail Fungus and other Nail Disease', 12), ('Poison Ivy Photos and other Contact Dermatitis', 13), ('Psoriasis pictures Lichen Planus and related diseases', 14), ('Scabies Lyme Disease and other Infestations and Bites', 15), ('Seborrheic Keratoses and other Benign Tumors', 16), ('Systemic Disease', 17), ('Tinea Ringworm Candidiasis and other Fungal Infections', 18), ('Urticaria Hives', 19), ('Vascular Tumors', 20), ('Vasculitis Photos', 21), ('Warts Molluscum and other Viral Infections', 22)]
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('./Xception/build')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()
 
st.write("""
         # Image Classification
         """
         )
 
file = st.file_uploader("Upload the image to be classified U0001F447", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)
 
def upload_predict(upload_image, model):
    
        size = (224,224)    
        image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img, dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        highest_pred_loc = np.argmax(prediction)
        
        return highest_pred_loc
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = upload_predict(image, model)
    diseaseName = classDict[predictions]
    st.write("The image is classified as",diseaseName[0])