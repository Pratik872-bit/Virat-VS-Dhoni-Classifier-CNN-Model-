import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# ================================================================
# STEP 1: Load Data (NO AUGMENTATION)
# ================================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 80% train, 20% validation
)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',        # Your dataset folder
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# ================================================================
# STEP 2: Build Transfer Learning Model
# ================================================================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150,150,3))
base_model.trainable = False  # Freeze base layers initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# ================================================================
# STEP 3: Compile Model
# ================================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ================================================================
# STEP 4: Callbacks (Early Stop + LR Scheduler)
# ================================================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# ================================================================
# STEP 5: Train the Model (Base Model Frozen)
# ================================================================


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# ================================================================
# STEP 6: Fine-Tune (Unfreeze last layers of base model)
# ================================================================
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Unfreeze only last 30 layers
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# ================================================================
# STEP 7: Save Model
# ================================================================
model.save('dhoni_vs_kohli_final_model.h5')
print("✅ Model saved successfully as 'dhoni_vs_kohli_final_model.h5'")

# ================================================================
# STEP 8: Plot Accuracy & Loss
# ================================================================
# ================================================================
# STEP 9: Test Prediction on New Image
# ================================================================

def predict_image(img_path):
    # Load image and preprocess
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0  # normalize
    pred = model.predict(x)[0][0]

    # Get class labels from training generator
    class_labels = list(train_generator.class_indices.keys())

    # Display result
    plt.imshow(img)
    plt.axis('off')
    if pred > 0.5:
        plt.title(f"🏏 Predicted: {class_labels[1]} (Probability: {pred:.2f})", color='orange')
        print(f"🏏 Predicted: {class_labels[1]} | Confidence: {pred:.2f}")
    else:
        plt.title(f"🏏 Predicted: {class_labels[0]} (Probability: {1-pred:.2f})", color='green')
        print(f"🏏 Predicted: {class_labels[0]} | Confidence: {1-pred:.2f}")
    plt.show()

# ================================================================
# STEP 10: Test on Custom Images
# ================================================================
# Example usage:
# Make sure image path is correct
# e.g., 'test_images/dhoni1.jpg' or 'test_images/kohli1.jpg'

test_image_path = 'test/testimage.jpg'   # 👈 change this to your test image path
if os.path.exists(test_image_path):
    predict_image(test_image_path)
else:
    print("⚠️ Image path not found. Please update 'test_image_path'.")
    
