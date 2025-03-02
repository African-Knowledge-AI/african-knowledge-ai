from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from .models.user import User
from .ai_database import get_db
from pydantic import BaseModel
import bcrypt

router = APIRouter()

# Pydantic Model for Registration
class UserRegister(BaseModel):
    email: str
    password: str

# Register User
@router.post("/register")
async def register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    try:
        # Hash the password
        hashed_password = bcrypt.hashpw(user_data.password.encode("utf-8"), bcrypt.gensalt()).decode()

        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")

        # Create user
        user = User(email=user_data.email, hashed_password=hashed_password)

        db.add(user)
        db.commit()
        db.refresh(user)

        return {"message": "User registered!", "api_key": user.api_key}
    
    except SQLAlchemyError as e:
        db.rollback()  # Rollback in case of database errors
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
