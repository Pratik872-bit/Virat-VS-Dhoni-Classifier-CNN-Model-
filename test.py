# ================================================================
# STEP 1: Import Required Libraries
# ================================================================
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# ================================================================
# STEP 2: Load Saved Model
# ================================================================
model = load_model('dhoni_vs_kohli_final_model.h5')
print("✅ Model loaded successfully!")

# ================================================================
# STEP 3: Define Prediction Function
# ================================================================
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"⚠️ Image not found: {img_path}")
        return

    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    pred = model.predict(x)[0][0]

    plt.imshow(img)
    plt.axis('off')

    if pred > 0.5:
        plt.title("🏏 Predicted: Virat Kohli", color='orange', fontsize=14)
        print(f"🏏 Predicted: Virat Kohli | Confidence: {pred:.2f}")
    else:
        plt.title("🏏 Predicted: MS Dhoni", color='green', fontsize=14)
        print(f"🏏 Predicted: MS Dhoni | Confidence: {1 - pred:.2f}")

    plt.show()

# ================================================================
# STEP 4: Test on Any New Image
# ================================================================
# 👇 Change this path to your actual test image path
test_image_path = r"test/new.jpg"

predict_image(test_image_path)
