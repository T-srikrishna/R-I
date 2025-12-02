# Thermal Emotion Recognition Project

**A deep learning system for recognizing emotions from thermal facial images**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-success.svg)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-88.00%25-brightgreen.svg)]()

## Project Overview

This project develops a Convolutional Neural Network (CNN) to classify emotions from thermal facial images. The model achieves **88.00% validation accuracy** across five emotion classes using thermal imaging data with multiple color palette representations.

### Key Highlights
- **88.00% Accuracy**: Production-ready baseline emotion recognition model
- **Multi-Palette Support**: Trained on 5 thermal color palettes per emotion (combinations of ICEBLUE, IRNBOW, IRON, RAINBOW, Red Hot, White Hot)
- **5 Emotion Classes**: Angry, Happy, Natural, Sad, Surprise
- **Lightweight**: Only 3.3M parameters, ~13MB model size
- **Fast**: Real-time inference capability
- **Production Ready**: Clean, tested, and documented codebase
- **Novel Insight**: Palette diversity provides superior augmentation compared to geometric transforms

---

## Repository Structure

```
R&I_ThermalCameras/
│
├── thermal_emotion_notebook.ipynb        # Main training notebook (PRODUCTION)
├── thermal_emotion_baseline_model.h5     # Trained baseline model (88.00% accuracy)
├── thermal_emotion_model_gridsearch.h5   # Grid search optimized (91.15% accuracy)
│
├── AccuracyEvaluation/                   # Real-world volunteer testing
│   ├── volunteerTesting.ipynb            # Volunteer video testing workflow
│   ├── thermal_emotion_model_finetuned.h5 # Fine-tuned on volunteer data (37.94%)
│   ├── Sub1/ through Sub6/               # Volunteer frame data by emotion
│   ├── finetune_data/                    # Train/val split for fine-tuning
│   │   ├── train/                        # Training frames (80%)
│   │   └── val/                          # Validation frames (20%)
│   ├── emotion_testing_results.csv       # Detailed performance metrics
│   ├── emotion_performance_summary.csv   # Per-emotion accuracy summary
│   └── model_comparison_summary.csv      # Before/after fine-tuning comparison
│
├── Facial emotion/                       # Training dataset (2,485 images)
│   ├── angry/                            # 6 thermal palettes each
│   ├── happy/
│   ├── natural/
│   ├── sad/
│   └── surprise/                         # Note: labeled as 'surpise' in filesystem
│
├── experimental/                         # PyTorch experiments
│   ├── thermal_emotion_pytorch.ipynb     # Experimental notebook
│   ├── thermal_resnet152_best.pth        # ResNet-152 model
│   ├── thermal_ensemble_model_*.pth      # 5 ensemble models
│   └── *.pth                             # Other experimental models
│
├── scripts/                              # Utility scripts
│   └── run_rtsp_detection.py             # Real-time RTSP thermal camera processing
│
├── documentation/                        # Project docs
│   ├── PROJECT_DOCUMENTATION.md          # Complete project documentation
│   └── (additional docs as needed)
│
├── presentations/                        # Meeting presentations
│   └── (presentation files)
│
└── README.md                             
```

---

## Quick Start

### Prerequisites

#### For Production Model (TensorFlow - CPU)
```bash
Python 3.13.3 (confirmed for TensorFlow 2.x compatibility)
TensorFlow 2.x (CPU version - GPU had compatibility issues)
OpenCV
NumPy, Pandas, Matplotlib, Scikit-learn
```

#### For Experimental Models (PyTorch - GPU Accelerated)
```bash
Python 3.11.6 (required for PyTorch GPU/CUDA compatibility)
PyTorch with CUDA 12.1 support (GPU acceleration)
OpenCV
NumPy, Pandas, Matplotlib, Scikit-learn
NVIDIA GPU with CUDA support (e.g., RTX 4060)
```

**Important Note on Python Versions & GPU Support:**
- **Production Notebook** (`thermal_emotion_notebook.ipynb`): Uses **Python 3.13.3** with TensorFlow 2.20.0
  - Trained on CPU (TensorFlow GPU had compatibility issues during development)
  
- **Experimental Notebook** (`experimental/thermal_emotion_pytorch.ipynb`): Uses **Python 3.11.6** with PyTorch 2.8.0
  - Switched to PyTorch specifically for **GPU acceleration** (CUDA support)
  - Python 3.11.6 + PyTorch 2.8.0+cu121 provides stable GPU training on Windows
  - This combination enabled faster experimental training (ResNet-152, ensemble models, etc.)

**Why Two Different Setups?**
- **TensorFlow GPU issues**: Encountered GPU compatibility problems with TensorFlow on Windows during initial development, so production model was trained on CPU
- **PyTorch GPU success**: Switched to PyTorch 2.8.0 with Python 3.11.6 for experimental work to leverage NVIDIA RTX 4060 GPU acceleration
- **Result**: Production model (CPU-trained) achieved 88.00%, experimental GPU-accelerated models reached 86.52%

If you're running both, you must use separate virtual environments for each framework.

### Installation

#### Option A: Production Environment (TensorFlow - Python 3.13.3)

```bash
# 1. Clone the repository
git clone https://github.com/T-srikrishna/R-I.git
cd R-I

# 2. Create virtual environment with Python 3.13.3
python3.13 -m venv .venv_tensorflow
.venv_tensorflow\Scripts\activate  # Windows
# or
source .venv_tensorflow/bin/activate  # Linux/Mac

# 3. Install TensorFlow dependencies
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn jupyter
```

#### Option B: Experimental Environment (PyTorch - Python 3.11.6)

```bash
# 1. Create separate virtual environment with Python 3.11.6
python3.11 -m venv .venv_pytorch
.venv_pytorch\Scripts\activate  # Windows
# or
source .venv_pytorch/bin/activate  # Linux/Mac

# 2. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy pandas matplotlib scikit-learn jupyter

# Note: PyTorch GPU (CUDA) requires Python 3.11.6 for Windows compatibility
# Python 3.13 has known GPU compatibility issues with PyTorch as of Oct 2025
```

**Why Two Environments?**
- **TensorFlow (production)**: Python 3.13.3, CPU training (GPU had compatibility issues)
- **PyTorch (experimental)**: Python 3.11.6 + CUDA 12.1, GPU acceleration with RTX 4060
- **Purpose**: PyTorch environment was created specifically to leverage GPU for faster experimental training
- Using separate environments prevents version conflicts and enables GPU acceleration for PyTorch experiments

### Usage

#### Option A: Use Pre-trained Model (Recommended)

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the production model
model = load_model('thermal_emotion_baseline_model.h5')

# Emotion classes (in order)
emotion_classes = ['angry', 'happy', 'natural', 'sad', 'surprise']

# Load and preprocess thermal image
img = cv2.imread('your_thermal_image.bmp')
img = cv2.resize(img, (128, 128))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

# Predict emotion
prediction = model.predict(img)
emotion_idx = np.argmax(prediction)
emotion = emotion_classes[emotion_idx]
confidence = np.max(prediction) * 100

print(f"Emotion: {emotion} ({confidence:.2f}% confidence)")
```

#### Option B: Train from Scratch

```bash
# Open the Jupyter notebook
jupyter notebook thermal_emotion_notebook.ipynb

# Run all cells to:
# - Load and preprocess data (2,485 images)
# - Train baseline CNN model (20 epochs)
# - Evaluate performance (88.00% accuracy)
# - Test augmentation hypothesis (palette vs geometric)
# - Save the trained model (thermal_emotion_baseline_model.h5)
```

---

## Model Performance

### Production Model: Baseline CNN (No Geometric Augmentation)

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **88.00%** |
| **F1 Score (Macro)** | 87.35% |
| **F1 Score (Weighted)** | 87.70% |
| **ROC AUC (Macro)** | 98.12% |
| **Parameters** | 3,305,285 |
| **Model Size** | ~12.61 MB |
| **Training Time** | ~88 seconds (20 epochs) |
| **Inference Time** | Real-time capable |

### Per-Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| **angry** | 0.94 | 0.94 | 0.94 | 98 |
| **happy** | 0.92 | 0.97 | 0.94 | 107 |
| **natural** | 0.94 | 0.86 | 0.90 | 102 |
| **sad** | 0.66 | 0.66 | 0.66 | 86 |
| **surprise** | 0.84 | 0.85 | 0.85 | 104 |

**Note**: Sad emotion has lower performance, likely due to similarity with natural expressions in thermal images.

### Architecture

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 emotions
])
```

### Training Configuration
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Regularization**: Dropout (0.5)
- **Data Augmentation**: None (baseline model - palette diversity is sufficient)
- **Batch Size**: 32
- **Epochs**: 20
- **Training Time**: ~88 seconds on GPU
- **Dataset Split**: 80% train (1,988 images), 20% validation (497 images)

### Key Finding: Palette Diversity > Geometric Augmentation

**Experimental Result**: Adding geometric augmentation (rotation, shift, zoom, flip) **decreased** performance by 45.27%

| Model | Accuracy | F1 (Macro) | Note |
|-------|----------|------------|------|
| **Baseline** (No geometric aug) | **88.00%** | **87.35%** | Best performance |
| **Augmented** (With geometric aug) | 40.64% | Lower | Performance degraded |

**Why Palette Diversity Works Better:**
- Dataset includes 6 color palettes per emotion (ICEBLUE, IRNBOW, IRON, RAINBOW, Red Hot, White Hot)
- Different palettes = natural color-based augmentation (6× effective data)
- Model learns color invariance from palettes
- Geometric transforms (rotation/flip) distort important thermal features
- **Conclusion**: Multiple thermal palettes provide superior augmentation compared to traditional geometric transforms

---

## Volunteer Testing & Real-World Validation

### Critical Discovery: Domain Gap Challenge

After achieving **91.15% validation accuracy** on the training dataset, we conducted real-world testing with live volunteer videos. This revealed a **critical gap between training performance and real-world deployment**.

### Testing Methodology

**Setup:**
- **Volunteers**: 6 participants (Sub1-Sub6)
- **Emotions Tested**: Angry, Happy, Natural, Sad, Surprise (5 emotions)
- **Video Collection**: Each volunteer recorded videos posing each emotion
- **Frame Extraction**: Every 5th frame extracted → 1,542 total test frames
- **Model Tested**: `thermal_emotion_model_gridsearch.h5` (91.15% validation accuracy)

**Process:**
1. Volunteers posed thermal emotion videos in controlled environment
2. Extracted frames from videos (every 5th frame to avoid redundancy)
3. Tested pre-trained model on extracted frames
4. Analyzed per-emotion and per-volunteer performance
5. Fine-tuned model on volunteer data
6. Compared original vs fine-tuned model performance

### Results Summary

| Metric | Original Model | Fine-Tuned Model | Improvement |
|--------|---------------|------------------|-------------|
| **Overall Accuracy** | **18.94%** | **37.94%** | **+19.00%** |
| Avg Confidence | 74.29% | 32.13% | -42.16% |
| Total Test Frames | 1,542 | 1,542 | - |
| Training Data Source | Comprehensive Facial Thermal Dataset | Volunteer Videos | Domain-Specific |

**Accuracy Drop**: 91.15% (validation) → 18.94% (volunteer videos) = **72.21% performance gap**

### Per-Emotion Performance (Original Model)

| Emotion | Accuracy | Frames Tested | Best Performing | Notes |
|---------|----------|---------------|-----------------|-------|
| Natural | 34.50% | 258 | ✓ | Best - neutral thermal patterns |
| Sad | 29.59% | 436 | | Second best |
| Surprise | 20.58% | 277 | | Moderate performance |
| Happy | 5.51% | 272 | | Struggled - confused with Sad |
| Angry | 0.67% | 299 | | Poorest - thermal signature mismatch |

### Key Findings

#### 1. Severe Domain Shift
- **Training Dataset**: 91.15% validation accuracy
- **Live Volunteer Videos**: 18.94% accuracy
- **Gap**: 72.21% performance drop due to domain shift

**Root Causes:**
- Different thermal camera specifications
- Different environment/lighting conditions  
- Natural expressions vs posed training data
- Thermal palette/color mapping differences
- Different thermal signature patterns between datasets

#### 2. Confidence Paradox
- **High confidence (74.29%)** despite **low accuracy (18.94%)**
- Model is confidently incorrect → overfitted to training domain
- Suggests training dataset patterns don't generalize to real-world footage

#### 3. Fine-Tuning Effectiveness
- **Doubled accuracy**: 18.94% → 37.94% (+19.00%)
- Demonstrates domain adaptation is essential
- Still below production requirements, but validates approach

#### 4. Emotion-Specific Challenges
- **Natural emotion**: Best performance (34.50%) - consistent thermal patterns
- **Angry emotion**: Worst performance (0.67%) - thermal signatures differ drastically
- **Model bias**: Overwhelming prediction toward "Sad" emotion
- **Class imbalance**: Predictions don't reflect true emotion distribution

### Implications for Production Deployment

⚠️ **WARNING: High validation accuracy ≠ Production readiness**

**Critical Lessons:**

1. **Domain-Specific Data is Essential**
   - Training data must match deployment environment
   - Collect data from same thermal camera model
   - Capture in same environmental conditions
   - Include natural (not just posed) expressions

2. **Fine-Tuning is Not Optional**
   - Pre-trained models require domain adaptation
   - Fine-tuning on target domain doubles performance
   - Budget time for collecting deployment-specific data

3. **Testing Protocol Matters**
   - Always validate on real deployment environment
   - Don't rely solely on validation set accuracy
   - Test with diverse volunteers and conditions

4. **Deployment Checklist**
   ```
   Before Production:
   ✓ Collect 500+ labeled frames from target thermal camera
   ✓ Fine-tune model on domain-specific data
   ✓ Validate on held-out target environment data  
   ✓ Achieve target accuracy (>80%) on real footage
   ✓ Implement confidence thresholding
   ✓ Set up continuous learning pipeline
   ```

### Fine-Tuning Details

**Training Configuration:**
- **Data Split**: 80% train (1,233 frames), 20% val (309 frames)
- **Augmentation**: Rotation ±10°, shift 10%, flip, zoom 10%
- **Epochs**: 30 with early stopping
- **Optimizer**: Adam (learning rate = 0.0001)
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

**Results:**
- **Validation Accuracy**: 41.36% (from 18.94%)
- **Test Accuracy**: 37.94%
- **Improvement**: +19.00% absolute gain
- **Status**: Better, but still needs more domain-specific training data

### Files Generated

**Notebook:**
- `AccuracyEvaluation/volunteerTesting.ipynb` - Complete testing workflow

**Models:**
- `AccuracyEvaluation/thermal_emotion_model_finetuned.h5` - Fine-tuned model

**Results:**
- `AccuracyEvaluation/emotion_testing_results.csv` - Detailed metrics
- `AccuracyEvaluation/emotion_performance_summary.csv` - Per-emotion breakdown
- `AccuracyEvaluation/model_comparison_summary.csv` - Before/after comparison

**Data:**
- `AccuracyEvaluation/Sub1/` through `Sub6/` - Extracted volunteer frames
- `AccuracyEvaluation/finetune_data/train/` - Training frames
- `AccuracyEvaluation/finetune_data/val/` - Validation frames

### Recommendations

**For Successful Production Deployment:**

1. **Allocate 2-4 weeks for domain-specific data collection**
   - Use target thermal camera in deployment environment
   - Collect 1,000+ labeled frames minimum
   - Include diverse subjects and lighting conditions

2. **Implement Continuous Learning**
   - Log low-confidence predictions during deployment
   - Periodically collect labeled production data
   - Retrain/fine-tune model quarterly
   - Monitor accuracy drift over time

3. **Hybrid Deployment Strategy**
   - Ensemble of training dataset + fine-tuned models
   - Confidence thresholding (reject predictions < 60%)
   - Temporal smoothing for video predictions
   - Fall back to simpler classification when uncertain

4. **Quality Assurance**
   - Maintain separate test set from target environment
   - Regular performance audits
   - A/B testing for model updates
   - User feedback collection

**Bottom Line**: A model with 91.15% validation accuracy achieved only 18.94% on real volunteer videos. **Always validate on your actual deployment environment before production.**

---

## Dataset

- **Total Images**: 2,485 thermal facial images
- **Emotion Classes**: 5 (angry, happy, natural, sad, surprise)
- **Thermal Palettes**: 5 palettes per emotion (different combinations: ICEBLUE, IRNBOW, IRON, RAINBOW, Red Hot, White Hot)
  - Note: Angry uses IRNBOW; Happy/Natural/Sad/Surprise use IRON
- **Original Image Size**: 320×240 pixels
- **Model Input Size**: 128×128 pixels (resized)
- **Image Format**: BMP files
- **Data Split**: 80% training (1,988 images), 20% validation (497 images)
- **Split Strategy**: Stratified by emotion to maintain class balance

---

## Experimental Work

The `experimental/` folder contains PyTorch-based research:

| Model | Accuracy | Parameters | Notes |
|-------|----------|------------|-------|
| **Ensemble (5 CNNs)** | 86.52% | 21.5M | Best PyTorch result |
| **Baseline CNN (PyTorch)** | 85.11% | 4.3M | Slightly lower than TensorFlow (88%) |
| **Transfer ResNet-50 v2** | 84.91% | 24.6M | Thermal-adapted conv1 |
| **Transfer ResNet-50 v1** | 67.00% | 24.6M | ImageNet features, overfitting |
| **Augmented CNN** | 29.78% | 4.3M | Geometric aug failed (-55%) |

**Key Finding**: Geometric augmentation fails in both frameworks (TensorFlow: -45%, PyTorch: -55%), confirming palette diversity is superior for thermal images.

See [`experimental/README.md`](experimental/README.md) for details.

---

## Documentation

Comprehensive documentation available:

- **[PROJECT_DOCUMENTATION.md](documentation/PROJECT_DOCUMENTATION.md)**: Complete project documentation and technical details
- Additional documentation files as needed

---

## Project Timeline

### Phase 1: Model Development (Weeks 1-6) - COMPLETE
- Week 1-2: Project setup and data collection
- Week 3: Data preprocessing and augmentation
- Week 4: Model training and debugging
- Week 5: Model evaluation and refinement
- Week 6: Code cleanup and documentation

### Phase 2: Real-World Deployment (Weeks 7-13) - IN PLANNING
- Week 7-8: Simulation environment development
- Week 9: Real-time processing pipeline
- Week 10: Face detection and temporal smoothing
- Week 11: Testing with simulated thermal data
- Weeks 12-13: Analysis and final presentation

**Note**: Original thermal camera (PRT-1217B6PA-TWBB) access unavailable. Developing simulation environment as contingency plan.

---

## Key Findings & Lessons Learned

### What Worked
1. **Simple CNN Architecture**: 3.3M parameters achieved 88% accuracy on 2,485-image dataset
2. **Palette Diversity**: 5 color palettes per emotion provide excellent natural augmentation
3. **No Geometric Augmentation**: Baseline without rotation/flip performs best
4. **Minimal Preprocessing**: Simple resize + normalization preserves thermal patterns
4. **Stratified Split**: Maintains emotion class balance across train/validation sets

### What Didn't Work
1. **Geometric Augmentation**: Rotation, shift, zoom, flip **decreased accuracy by 45.27%**
2. **Over-Augmentation**: Traditional CV augmentation techniques distort thermal features
3. **Aggressive Transforms**: Thermal emotion patterns are spatially sensitive

### Lessons Learned
1. **Domain-Specific Approach**: Thermal images ≠ RGB images - require different augmentation strategies
2. **Palette > Geometry**: Color palette diversity provides better augmentation than geometric transforms
3. **Natural Augmentation**: Dataset design with multiple palettes creates built-in robustness
4. **Simplicity Wins**: Baseline model without geometric augmentation achieves best results
5. **Thermal Features**: Spatial thermal patterns must be preserved - avoid distorting transforms

### Novel Research Finding
**Discovery**: Multi-palette thermal datasets provide superior augmentation compared to traditional geometric transforms. Baseline model with palette diversity achieved 88% accuracy without any geometric augmentation, while adding geometric transforms decreased performance to 40.64% (-47.36% drop).

**Impact**: This finding suggests thermal emotion recognition benefits more from color invariance training (via multiple palettes) than spatial augmentation. Future thermal datasets should prioritize palette diversity over geometric variation.

---

## Team

- **Ajju Dangol** - Project Lead & Systems Integrator
- **Prem Prasad Bhatta** - Data Scientist
- **Srikrishna Thapa** - AI Researcher & Documentation
- **Abhishek Abhishek** - AI Researcher & Facilitator

---

## Future Work

### Phase 2 Plans
- [ ] Real-time thermal camera integration
- [ ] Multi-person detection and tracking
- [ ] Temporal smoothing for stable predictions
- [x] Live volunteer testing
- [ ] Performance optimization for real-time processing

### Research Extensions
- [ ] Investigate why "sad" emotion has lower accuracy (66% vs 84-94% for other emotions)
- [ ] Test with additional thermal palettes for even better color invariance
- [ ] Explore attention mechanisms to focus on key thermal facial regions
- [ ] Multi-modal fusion (thermal + RGB) for enhanced accuracy
- [ ] Temporal analysis for emotion dynamics in video sequences
- [ ] Cross-dataset validation with other thermal emotion databases

---

## Contact & Links

- **Repository**: [github.com/T-srikrishna/R-I](https://github.com/T-srikrishna/R-I)
- **Presentations**: See [`presentations/`](presentations/) folder
- **Documentation**: See [`documentation/`](documentation/) folder

---

## License

This project is part of academic research. All rights reserved.

---

## Acknowledgments

- Research team members for collaboration and support
- TensorFlow and PyTorch communities
- Comprehensive Facial Thermal Dataset (DOI: 10.17632/8885sc9p4z.1)

---

**Last Updated**: December 2, 2025  
**Version**: 1.0.0 (Phase 1 Complete)  
**Status**: Production-Ready Model | Phase 2 Planning
