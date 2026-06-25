# Handwritten Digit Recognizer

A CNN-based Handwritten Digit Recognizer trained on the MNIST dataset. This project uses TensorFlow and Keras to classify handwritten digits (0–9) with high accuracy and provides an interactive web interface using Streamlit.

## Features

* CNN model trained on the MNIST dataset
* Real-time handwritten digit prediction
* Interactive Streamlit web application
* Training accuracy and loss visualization
* Sample prediction images

## Tech Stack

* Python
* TensorFlow
* Keras
* Streamlit
* NumPy
* Matplotlib

## Project Structure

```
digit_recognizer/
│
├── app.py                     # Streamlit application
├── train_model.py             # CNN model training script
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── model/
    ├── digit_model.h5         # Trained CNN model
    ├── training_plot.png      # Training graphs
    └── sample_predictions.png # Sample predictions
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/saurabhsakhare955-ship-it/digit_recognizer.git
cd digit_recognizer
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

The application will open in your browser, allowing you to draw or upload handwritten digits and obtain predictions instantly.

## Model Performance

* Dataset: MNIST
* Architecture: Convolutional Neural Network (CNN)
* Accuracy: ~99%

## Future Improvements

* Support custom image uploads
* Improve UI/UX
* Deploy using Streamlit Cloud

## Author

**Saurabh Sakhare**

B.Tech CSE, MIT ADT University
