# 🌡️ Thermal Emotion Recognition - Laptop Camera Testing

## 🎯 Simple Testing Setup

This simplified version lets you test thermal emotion recognition using your **laptop camera** without needing actual thermal camera hardware.

### ✅ What You Have

- **Enhanced Emotion Model**: 88.93% accuracy (trained on thermal images)
- **Basic Emotion Detection**: Regular laptop camera emotion recognition
- **Thermal Camera Simulator**: Makes your laptop camera look like a thermal camera
- **Temperature Simulation**: Simulates body temperature based on face brightness

### 🚀 Quick Start

#### Option 1: Easy Launcher
```bash
python run_thermal_test.py
```
Then choose:
- **Mode 1**: Basic emotion detection
- **Mode 2**: Thermal camera simulator (recommended!)

#### Option 2: Direct Run
```bash
# Basic emotion detection
python live_emotion_camera.py

# Thermal simulator (cooler!)
python thermal_simulator.py
```

### 🎮 Controls

#### Basic Mode:
- `q` - Quit

#### Thermal Simulator Mode:
- `q` - Quit
- `s` - Save screenshot
- `e` - Export data to JSON
- `r` - Reset data
- `SPACE` - Pause/Resume

### 🌡️ Thermal Simulator Features

The thermal simulator makes your regular camera look like a real thermal camera:

- **🎨 Thermal Visual Effects**: Applies blue-to-red thermal colormap
- **🌡️ Temperature Simulation**: Estimates "body temperature" from face brightness
- **🧠 Emotion Detection**: Uses your trained 88.93% accuracy model
- **📊 Real-time Analytics**: Shows average emotion and temperature
- **⚠️ Temperature Alerts**: Warnings for high/low temperatures
- **📈 Data Logging**: Saves all detections for analysis

### 📊 What You'll See

```
🎭 Emotion: happy (0.92)
🌡️ Temperature: 36.8°C
📏 Bounding box around face
📊 FPS counter
📈 Average emotion over time
🔄 Detection count
```

### 🔧 How Temperature Simulation Works

1. **Face Detection**: Finds your face using Haar cascades
2. **Brightness Analysis**: Measures average brightness of face region
3. **Temperature Mapping**: Converts brightness to simulated temperature
   - Brighter areas = warmer (like real thermal cameras)
   - Range: 34-40°C (realistic body temperature range)
4. **Noise Addition**: Adds realistic temperature variations

### 📸 Screenshots and Data

- **Screenshots**: Saved as `thermal_sim_screenshot_YYYYMMDD_HHMMSS.jpg`
- **Data Export**: Saved as `thermal_sim_data_YYYYMMDD_HHMMSS.json`

Example exported data:
```json
[
  {
    "bbox": [100, 50, 200, 200],
    "emotion": "happy",
    "confidence": 0.92,
    "temperature": 36.8,
    "timestamp": "2025-09-22T21:00:00"
  }
]
```

### 🎯 Perfect for Testing

This setup is ideal for:
- **🧪 Testing your emotion model** without thermal hardware
- **🎨 Seeing how thermal cameras work** visually
- **📊 Understanding the data structure** for real thermal integration
- **🎮 Demonstrating the system** to others
- **🔧 Developing features** before buying thermal cameras

### 🔄 Next Steps

When you're ready for real thermal cameras:
1. Choose a thermal camera model (FLIR, Seek Thermal, etc.)
2. Replace the simulation with real thermal camera drivers
3. Calibrate temperature readings for accuracy
4. Keep the same emotion detection pipeline!

### ⚠️ Notes

- Temperature values are **simulated** (not real body temperature)
- Emotion detection uses your real trained model (88.93% accuracy)
- Works best in good lighting conditions
- Face detection requires clear frontal face view

---

## 🚀 Start Testing Now!

```bash
python run_thermal_test.py
```

Choose mode 2 (Thermal Simulator) for the full experience! 🌡️🎭