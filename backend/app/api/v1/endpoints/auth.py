from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel
from passlib.context import CryptContext
from datetime import datetime, timedelta
from app.core.utils import authenticate_user, create_access_token

router = APIRouter()

# OAuth2PasswordBearer allows the client to send the token in the request header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class User(BaseModel):
    username: str


class UserInDB(User):
    hashed_password: str


# Token endpoint to generate JWT token
@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


# Example protected route
@router.get("/secure-data")
async def get_secure_data(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, "your_secret_key", algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"message": "This is secure data!", "user": username}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
