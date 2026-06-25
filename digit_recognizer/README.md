# 🧠 Handwritten Digit Recognizer

A CNN-based handwritten digit classifier trained on the MNIST dataset achieving ~99% accuracy.

## 🛠️ Tech Stack
- **Python 3.8+**
- **TensorFlow / Keras** – CNN model
- **Streamlit** – Web application
- **NumPy, Matplotlib, Pillow** – Data processing & visualization

## 📁 Project Structure
```
digit_recognizer/
├── train_model.py        # Train the CNN model
├── app.py                # Streamlit web application
├── requirements.txt      # Python dependencies
└── model/                # Auto-created after training
    ├── digit_model.h5        # Saved model
    ├── training_plot.png     # Accuracy/Loss curves
    └── sample_predictions.png
```

## 🚀 Setup & Run

### Step 1: Install Python
Download from https://www.python.org/downloads/ (Python 3.8 or higher)

### Step 2: Install dependencies
Open terminal/command prompt in this folder and run:
```bash
pip install -r requirements.txt
```

### Step 3: Train the model
```bash
python train_model.py
```
This downloads MNIST (~11MB) and trains for 10 epochs (~5-10 mins).
Expected accuracy: **~99%**

### Step 4: Run the web app
```bash
streamlit run app.py
```
Opens at: http://localhost:8501

## 🎯 How to Use
1. Run the app with `streamlit run app.py`
2. Upload a handwritten digit image (PNG or JPG)
3. The model predicts the digit with confidence score
4. View probability distribution for all digits (0-9)

## 🧠 Model Architecture
```
Input (28×28×1)
    → Conv2D(32) + BatchNorm + MaxPool
    → Conv2D(64) + BatchNorm + MaxPool
    → Conv2D(64) + BatchNorm
    → Flatten
    → Dense(128) + Dropout(0.3)
    → Output Dense(10) + Softmax
```

## 📊 Results
- Training Accuracy: ~99.5%
- Test Accuracy: ~99%
- Dataset: MNIST (60,000 train / 10,000 test)

## 💡 Jury Talking Points
- CNNs capture spatial features (edges, curves) unlike flat classifiers
- BatchNormalization speeds training and improves stability
- Dropout prevents overfitting
- MNIST is the standard benchmark for digit recognition
