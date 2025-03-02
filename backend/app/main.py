
from fastapi import FastAPI
from app.api.v1.endpoints import auth, ai, doc_ai

from app.api.v1 import auth



app = FastAPI()



# Include routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(ai.router, prefix="/ai", tags=["ai"])
app.include_router(doc_ai.router, prefix="/documents", tags=["documents"])
app.include_router(auth.router, prefix="/users", tags=["users"]) 

@app.get("/")
async def root():
    return {"message": "Welcome to the backend API!"}



