from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import numpy as np

# Folder path jahan Dhoni ke images hain
img_dir = 'dataset/train/virat_kohli'

# Image augmentation setup
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Target number of total images (example: you want ~1000 total)

target_count = 2000
current_count = len(os.listdir(img_dir))
needed = target_count - current_count

print(f"🟡 Current images: {current_count}")
print(f"🟢 Need to generate: {needed} new images")

generated = 0

# Loop through existing images
for filename in os.listdir(img_dir):
    if generated >= needed:
        break  # Stop once we generated enough

    img_path = os.path.join(img_dir, filename)
    img = load_img(img_path)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Generate and save augmented images directly in same folder
    i = 0
    for batch in datagen.flow(
        x,
        batch_size=1,
        save_to_dir=img_dir,         # ✅ Same folder
        save_prefix='aug_virat',     # Prefix for new images
        save_format='jpg'            # File format
    ):
        i += 1
        generated += 1
        if i > 1 or generated >= needed:  # 1 new image per original
            break

print(f"✅ Done! Total images in folder now: {len(os.listdir(img_dir))}")
