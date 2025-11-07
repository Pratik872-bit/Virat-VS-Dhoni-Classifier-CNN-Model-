import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ==============================
# STEP 1: Data Loading
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values
    validation_split=0.2    # 20% for validation
)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    'dataset/train',        # Path to your dataset folder
    target_size=(150, 150), # Resize all images
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation data generator
val_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# ==============================
# STEP 2: Build CNN Model
# ==============================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),     # reduce overfitting
    layers.Dense(1, activation='sigmoid')  # binary classification (Dhoni/Kohli)
])

# ==============================
# STEP 3: Compile the Model
# ==============================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==============================
# STEP 4: Train the Model
# ==============================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    verbose=1
)

# ==============================
# STEP 5: Plot Accuracy & Loss
# ==============================
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# ==============================
# STEP 6: Save Model
# ==============================
model.save('dhoni_vs_kohli_model.h5')
print("✅ Model saved successfully as 'dhoni_vs_kohli_model.h5'")
