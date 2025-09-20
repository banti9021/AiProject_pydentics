from pydantic import BaseModel

class InputData(BaseModel):
    age: int
    bmi: float
    children: int
    sex: int
    region: int

class PredictionResponse(BaseModel):
    id: int
    age: int
    bmi: float
    children: int
    sex: int
    region: int
    is_smoker: bool

    class Config:
        orm_mode = True
