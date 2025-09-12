import os
from collections import defaultdict

# Path to the main dataset folder
DATASET_DIR = r"d:\R&I\Facial emotion"

# Dictionary to hold counts
counts = defaultdict(lambda: defaultdict(int))

for emotion in os.listdir(DATASET_DIR):
    emotion_path = os.path.join(DATASET_DIR, emotion)
    if not os.path.isdir(emotion_path):
        continue
    for palette in os.listdir(emotion_path):
        palette_path = os.path.join(emotion_path, palette)
        if not os.path.isdir(palette_path):
            continue
        image_files = [f for f in os.listdir(palette_path) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))]
        counts[emotion][palette] = len(image_files)

# Print results
print("Image counts per emotion and palette:")
for emotion, palettes in counts.items():
    print(f"Emotion: {emotion}")
    for palette, count in palettes.items():
        print(f"  Palette: {palette} - {count} images")
    print()
