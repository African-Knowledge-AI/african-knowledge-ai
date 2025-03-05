import secrets
from sqlalchemy import Column, String
from app.api.v1.ai_database import Base

#user model
class User(Base):
    __tablename__ = "users"

    email = Column(String, primary_key=True, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    api_key = Column(String, unique=True, index=True, default=lambda: secrets.token_hex(16))
