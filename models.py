from sqlalchemy import Column, Integer, Boolean
from database import Base

class PredictionDB(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer, nullable=False)
    bmi = Column(Integer, nullable=False)
    children = Column(Integer, nullable=False)
    sex = Column(Integer, nullable=False)
    region = Column(Integer, nullable=False)
    is_smoker = Column(Boolean, nullable=False)
