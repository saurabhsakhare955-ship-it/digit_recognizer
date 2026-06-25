import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 50)
print("  Handwritten Digit Recognizer - Training")
print("=" * 50)

# Load MNIST dataset
print("\n[1/4] Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(f"  Training samples: {len(x_train)}")
print(f"  Test samples:     {len(x_test)}")

# Preprocess
print("\n[2/4] Preprocessing data...")
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat  = tf.keras.utils.to_categorical(y_test, 10)

# Build CNN
print("\n[3/4] Building CNN model...")
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train
print("\n[4/4] Training model (10 epochs)...")
history = model.fit(
    x_train, y_train_cat,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\n{'='*50}")
print(f"  ✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"  ✅ Test Loss:     {test_loss:.4f}")
print(f"{'='*50}")

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/digit_model.h5")
print("\n✅ Model saved to model/digit_model.h5")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('CNN Training Results - Handwritten Digit Recognizer', fontsize=14)

axes[0].plot(history.history['accuracy'], 'b-o', label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], 'r-o', label='Val Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'], 'b-o', label='Train Loss')
axes[1].plot(history.history['val_loss'], 'r-o', label='Val Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("model/training_plot.png", dpi=150, bbox_inches='tight')
print("✅ Training plot saved to model/training_plot.png")

# Show sample predictions
print("\n✅ Showing sample predictions...")
fig2, axes2 = plt.subplots(2, 5, figsize=(12, 5))
fig2.suptitle('Sample Predictions on Test Set', fontsize=14)
indices = np.random.choice(len(x_test), 10, replace=False)

for i, idx in enumerate(indices):
    ax = axes2[i // 5][i % 5]
    pred = model.predict(x_test[idx:idx+1], verbose=0)
    predicted = np.argmax(pred)
    actual = y_test[idx]
    color = 'green' if predicted == actual else 'red'
    ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f'Pred: {predicted}\nActual: {actual}', color=color, fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig("model/sample_predictions.png", dpi=150, bbox_inches='tight')
print("✅ Sample predictions saved to model/sample_predictions.png")
print("\n🎉 Training complete! Now run: streamlit run app.py")
