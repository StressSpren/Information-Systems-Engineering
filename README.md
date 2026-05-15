# OCR System for Suffolk Archives

## Overview
This is a Python-based Optical Character Recognition (OCR) project for Suffolk Archives that uses deep learning models to recognize handwritten digits. The project includes code for an AWS EC2 server for training, prediction and model distributuion, and a PyQt6 GUI for client interaction.

## Features
- Train deep learning models (CNN) on the MNIST and DIDA datasets
- GUI built with PyQt6 for batch or singular image upload and prediction
- REST API for model distribution using Flask
- AWS EC2 server integration
- AWS S3 integration (upload/download images)

## Project Structure
```
Implementation/
		Client/
				main.py                # GUI application
				functions/             # Function modules
				models/                # Trained models
				predicted_images/      # Labelled output predictions
		EC2/
				train_model.py         # Model training script
				FLASK-GEN/app.py       # Flask API server
				models/                # Trained models files for EC2
				datasets/              # Training datasets
models/                        # Saved models
notebooks/                     # Classification model notebook
predicted_images/              # Output image predictions 
requirements.txt
test_images/                   # Images for testing
Implementation.mp4  		   # Video demonstrating application
```

## Usage

### REST API
- Start the Flask server on the EC2 server:
	```bash
	python Implementation/EC2/FLASK-GEN/app.py
	```
### Training
- Run the training script to train a model:
	```bash
	python Implementation/EC2/train_model.py
	```
## Testing
- Run the `notebooks/CNN.ipynb` to see the CNNs functionality 

### Prediction (GUI)
- Launch the GUI application on the clients machine:
	```bash
	python Implementation/Client/main.py
	```

## Datasets
- MNIST and DIDA datasets are stored in `datasets/` and `Implementation/EC2/datasets/`.

## Requirements
See `requirements.txt` for all imports
