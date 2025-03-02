import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Securely load environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
