import uvicorn
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import base64
from pydantic import BaseModel

# from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "exp://192.168.1.7:19000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
classDict = [
    ("Acne and Rosacea Photos", 0),
    ("Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions", 1),
    ("Atopic Dermatitis Photos", 2),
    ("Bullous Disease Photos", 3),
    ("Cellulitis Impetigo and other Bacterial Infections", 4),
    ("Eczema Photos", 5),
    ("Exanthems and Drug Eruptions", 6),
    ("Hair Loss Photos Alopecia and other Hair Diseases", 7),
    ("Herpes HPV and other STDs Photos", 8),
    ("Light Diseases and Disorders of Pigmentation", 9),
    ("Lupus and other Connective Tissue diseases", 10),
    ("Melanoma Skin Cancer Nevi and Moles", 11),
    ("Nail Fungus and other Nail Disease", 12),
    ("Poison Ivy Photos and other Contact Dermatitis", 13),
    ("Psoriasis pictures Lichen Planus and related diseases", 14),
    ("Scabies Lyme Disease and other Infestations and Bites", 15),
    ("Seborrheic Keratoses and other Benign Tumors", 16),
    ("Systemic Disease", 17),
    ("Tinea Ringworm Candidiasis and other Fungal Infections", 18),
    ("Urticaria Hives", 19),
    ("Vascular Tumors", 20),
    ("Vasculitis Photos", 21),
    ("Warts Molluscum and other Viral Infections", 22),
]

model = None


class Item(BaseModel):
    data: str


def load_model():
    model = tf.keras.models.load_model("./Xception/build")
    print("Model loaded")
    return model


def read_imagefile(file) -> Image.Image:
    decoded = base64.b64decode(file)
    image = Image.open(BytesIO(decoded))
    # image.show()
    return image


def upload_predict(upload_image):
    global model
    if model is None:
        model = load_model()
    size = (224, 224)
    image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    img_reshape = img_resize[np.newaxis, ...]
    with tf.device("/cpu:0"):
        prediction = model.predict(img_reshape)
    highest_pred_loc = np.argmax(prediction)

    return highest_pred_loc


@app.post("/predict/image")
async def predict_api(item: Item):
    # print(item.data)
    if not item.data:
        return {"response": "Error File Not received"}

    image = read_imagefile(item.data)
    predictions = upload_predict(image)
    return classDict[predictions]
    # return {"filename": file.filename}


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
