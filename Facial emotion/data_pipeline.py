import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Path to dataset
DATASET_DIR = r"d:\R&I\Facial emotion"
IMG_SIZE = (128, 128)  # You can adjust this size
BATCH_SIZE = 32

# Prepare lists for images and labels
def get_image_paths_and_labels(dataset_dir):
    image_paths = []
    emotion_labels = []
    palette_labels = []
    for emotion in os.listdir(dataset_dir):
        emotion_path = os.path.join(dataset_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue
        for palette in os.listdir(emotion_path):
            palette_path = os.path.join(emotion_path, palette)
            if not os.path.isdir(palette_path):
                continue
            for fname in os.listdir(palette_path):
                if fname.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(palette_path, fname))
                    emotion_labels.append(emotion)
                    palette_labels.append(palette)
    return image_paths, emotion_labels, palette_labels

image_paths, emotion_labels, palette_labels = get_image_paths_and_labels(DATASET_DIR)

# Encode labels
from sklearn.preprocessing import LabelEncoder
emotion_encoder = LabelEncoder()
palette_encoder = LabelEncoder()
emotion_labels_encoded = emotion_encoder.fit_transform(emotion_labels)
palette_labels_encoded = palette_encoder.fit_transform(palette_labels)

# Split into train/val
train_idx, val_idx = train_test_split(
    np.arange(len(image_paths)), test_size=0.2, stratify=emotion_labels_encoded, random_state=42
)

train_paths = [image_paths[i] for i in train_idx]
train_emotions = [emotion_labels_encoded[i] for i in train_idx]
train_palettes = [palette_labels_encoded[i] for i in train_idx]

val_paths = [image_paths[i] for i in val_idx]
val_emotions = [emotion_labels_encoded[i] for i in val_idx]
val_palettes = [palette_labels_encoded[i] for i in val_idx]

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

# Helper function to generate batches
def data_generator(paths, emotions, palettes, datagen):
    while True:
        idxs = np.random.permutation(len(paths))
        for i in range(0, len(paths), BATCH_SIZE):
            batch_idxs = idxs[i:i+BATCH_SIZE]
            batch_images = []
            batch_emotions = []
            batch_palettes = []
            for j in batch_idxs:
                img = datagen.flow_from_directory(
                    os.path.dirname(paths[j]),
                    target_size=IMG_SIZE,
                    batch_size=1,
                    class_mode=None,
                    shuffle=False
                ).next()[0]
                batch_images.append(img)
                batch_emotions.append(emotions[j])
                batch_palettes.append(palettes[j])
            yield np.array(batch_images), np.array(batch_emotions), np.array(batch_palettes)

# Example usage:
# train_gen = data_generator(train_paths, train_emotions, train_palettes, train_datagen)
# val_gen = data_generator(val_paths, val_emotions, val_palettes, val_datagen)

print(f"Total images: {len(image_paths)}")
print(f"Train images: {len(train_paths)}")
print(f"Validation images: {len(val_paths)}")
print(f"Emotion classes: {emotion_encoder.classes_}")
print(f"Palette classes: {palette_encoder.classes_}")
