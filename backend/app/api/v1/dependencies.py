from fastapi import Depends, HTTPException, Header
from sqlalchemy.orm import Session
from .models.user import User

async def verify_api_key(x_api_key: str = Header(None), db: Session = Depends(lambda: None)):
    from app.api.v1.ai_database import get_db  # Import inside function to break circular dependency
    db = get_db()

    if not x_api_key:
        raise HTTPException(status_code=403, detail="API key missing")

    user = db.query(User).filter(User.api_key == x_api_key).first()
    if not user:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return user
