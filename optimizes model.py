import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

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

model.save('dhoni_vs_kohli_final_model.h5')
print("✅ Model saved successfully as 'dhoni_vs_kohli_final_model.h5'")

# ================================================================
# STEP 9: Test Prediction on New Image
# ================================================================







# ================================================================
# STEP 7: Plot Accuracy & Loss
# ================================================================
# plt.figure(figsize=(8,5))
# plt.plot(history.history['accuracy'] + fine_tune_history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'], label='Validation Accuracy')
# plt.legend()
# plt.title('Model Accuracy')
# plt.show()

# plt.figure(figsize=(8,5))
# plt.plot(history.history['loss'] + fine_tune_history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'] + fine_tune_history.history['val_loss'], label='Validation Loss')
# plt.legend()
# plt.title('Model Loss')
# plt.show()