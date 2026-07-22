from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os

from app.predict import predict

app = FastAPI(
    title="Quran Reciter Classification API",
    version="1.0"
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "Quran Reciter Classification API is Running!"
    }


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """
    Predict the reciter from an uploaded audio file.
    """

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict(file_path)

    os.remove(file_path)

    return JSONResponse(result)