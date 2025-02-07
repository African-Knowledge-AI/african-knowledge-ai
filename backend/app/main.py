from fastapi import FastAPI
from app.api.v1.endpoints import auth, ai

app = FastAPI()

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(ai.router, prefix="/ai", tags=["ai"])

@app.get("/")
async def root():
    return {"message": "Welcome to the backend API!"}
