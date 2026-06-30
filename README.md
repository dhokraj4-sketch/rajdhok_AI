# MNIST Handwritten Digit Classification

This is a moderate college-submission project for recognizing handwritten digits from 0 to 9 using a neural network trained on the MNIST dataset.

## Project Features

- Loads the MNIST handwritten digit dataset
- Preprocesses images by scaling pixel values from 0-255 to 0-1
- Trains a simple neural network using TensorFlow/Keras
- Evaluates model accuracy on test data
- Saves the trained model for reuse
- Predicts a digit from a custom image such as `3-digit.PNG`
- Includes an optional Streamlit web app

## Folder Structure

```text
College_Project_MNIST/
  app.py
  requirements.txt
  README.md
  3-digit.PNG
  MNIST_Working_Project.ipynb
  src/
    __init__.py
    preprocess.py
    train.py
    predict.py
```

Generated after training:

```text
models/
  mnist_model.keras
outputs/
  accuracy_plot.png
  confusion_matrix.png
```

## Setup

Create and activate a virtual environment, then install the required packages.

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python -m src.train
```

This trains the model and saves it at:

```text
models/mnist_model.keras
```

## Predict the Sample Image

```bash
python -m src.predict 3-digit.PNG
```

Expected result:

```text
Predicted digit: 3
```

## Run the Web App

```bash
streamlit run app.py
```

Upload a handwritten digit image and the app will show the predicted digit with confidence scores.

## Project Summary

MNIST is a standard dataset containing 70,000 grayscale images of handwritten digits. Each image is 28 x 28 pixels. The model used in this project is a feed-forward neural network with dense layers. It is simple enough for a college project while still giving high accuracy on the test dataset.
