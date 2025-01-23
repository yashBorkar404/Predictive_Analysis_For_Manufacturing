import pandas as pd
from fastapi import UploadFile

async def save_uploaded_file(file: UploadFile, destination: str):
    with open(destination, "wb") as buffer:
        buffer.write(await file.read())

def load_data(filepath: str):
    return pd.read_csv(filepath)