from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.api.v1.config import DATABASE_URL


# Create database engine
engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Dependency for getting the DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
