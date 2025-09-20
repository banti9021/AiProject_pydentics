from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
import joblib
import numpy as np

import models, schemas
from database import SessionLocal, engine

# Create DB tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Load model and scaler
model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/predict", response_model=schemas.PredictionResponse)
def predict(data: schemas.InputData, db: Session = Depends(get_db)):
    # Prepare data
    input_array = np.array([[data.age, data.bmi, data.children, data.sex, data.region]])
    input_scaled = scaler.transform(input_array)

    # Predict
    pred = model.predict(input_scaled)[0]
    is_smoker = bool(pred)

    # Save to DB
    db_record = models.PredictionDB(
        age=data.age,
        bmi=data.bmi,
        children=data.children,
        sex=data.sex,
        region=data.region,
        is_smoker=is_smoker
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return db_record

@app.get("/predictions")
def get_predictions(db: Session = Depends(get_db)):
    return db.query(models.PredictionDB).all()
