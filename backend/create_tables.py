from sqlalchemy import create_engine
from app.api.v1.ai_database import Base  # Import your Base class
from app.api.v1.config import DATABASE_URL

# Create engine
engine = create_engine(DATABASE_URL)

# Create all tables
Base.metadata.create_all(bind=engine)

print("✅ Tables created successfully!")
