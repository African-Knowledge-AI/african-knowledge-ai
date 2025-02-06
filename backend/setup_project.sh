#!/bin/bash

# Set the backend directory
BACKEND_DIR="/workspaces/african-knowledge-ai/backend"

# Create directories
mkdir -p $BACKEND_DIR/app/{models,api/v1/endpoints,core}

# Create files
touch $BACKEND_DIR/app/main.py
touch $BACKEND_DIR/app/api/v1/endpoints/{auth.py,ai.py}
touch $BACKEND_DIR/app/core/{config.py,utils.py}
touch $BACKEND_DIR/app/models/user.py
touch $BACKEND_DIR/requirements.txt

echo "Project structure created successfully!"
