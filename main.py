from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = FastAPI()

# Load the model
model = load_model('riceplantdetectionmodel.h5')

# Define class names
class_names = ['Bacterialblight', 'Brownspot', 'Leafsmut']

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))
    
    # Resize and preprocess the image
    img = img.resize((300, 300))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    prediction = model.predict(img_array)[0]
    class_label = np.argmax(prediction)

    # Get result
    result = class_names[class_label]

    return JSONResponse(content={"result": result})

# To run the application, use the command below
# uvicorn main:app --reload
