from fastapi import Depends, HTTPException, Header
from sqlalchemy.orm import Session
from app.api.v1.ai_database import get_db  # Ensure this returns a generator
from .models.user import User

async def verify_api_key(
    x_api_key: str = Header(None),
    db: Session = Depends(get_db)  # Correctly handle the dependency
):
    if not x_api_key:
        raise HTTPException(status_code=403, detail="API key missing")

    # Convert generator to session
    if isinstance(db, type(get_db())):  # If db is a generator, extract the session
        db = next(db)

    user = db.query(User).filter(User.api_key == x_api_key).first()
    if not user:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return user
