#!/bin/bash

# Check if the model file exists
if [ ! -f "short-phrase-generation.pt" ]; then
    echo "Model file not found. Training the model..."
    python train.py data/shakespeare.txt --hidden-size=200
else
    echo "Model file found. Skipping training."
fi

# Start the Flask application
python app.py