import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Define the SQLite database URL
DATABASE_URL = "sqlite:///./medical_ai.db"

# 2. Configure the SQLAlchemy Engine
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}  # Crucial for multithreaded apps like Streamlit/FastAPI
)

# 3. Create Session Local generator factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. Create declarative base class for mapping models
Base = declarative_base()

# 5. Define the missing PredictionHistory database model table
class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    diagnosis = Column(String)
    confidence = Column(Float)
    report = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# 6. Session database dependency injector helper
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()