# Experimental PyTorch Models

This folder contains experimental PyTorch-based models and research for thermal emotion recognition.

## Important: Python Version Requirement

**This experimental notebook requires Python 3.11.6 for GPU acceleration.**

### Why Python 3.11.6?
- **GPU Compatibility**: PyTorch 2.8.0 with CUDA support has compatibility issues with Python 3.13 on Windows (as of October 2025)
- **CUDA Support**: Python 3.11.6 provides stable CUDA 12.1 support for PyTorch training on Windows
- **RTX 4060 Optimization**: Our training uses NVIDIA RTX 4060 Laptop GPU which requires PyTorch 2.8.0+cu121 built for Python 3.11.x

### Python Version Differences in This Project
| Environment | Python Version | Framework | Location | Reason |
|-------------|---------------|-----------|----------|--------|
| **Production** | Python 3.13.3 | TensorFlow 2.20.0 | Root notebook | Full TensorFlow GPU support on Windows |
| **Experimental** | Python 3.11.6 | PyTorch 2.8.0+cu121 | experimental/ | PyTorch CUDA GPU compatibility requirement |

## Contents

### Notebook
- **thermal_emotion_pytorch.ipynb** - Experimental PyTorch notebook with various architectures

### Models (Generated after running notebook)

#### Ensemble Models
- `thermal_ensemble_model_1.pth` through `_5.pth` - 5 ensemble CNN models (86.52% combined accuracy)
- `ensemble_info.pkl` - Ensemble metadata

#### Transfer Learning Models
- `thermal_transfer_resnet50_best.pth` - ResNet-50 transfer learning v1 (67.00%)
- `thermal_transfer_v2_resnet50_best.pth` - ResNet-50 thermal-adapted v2 (84.91%)

#### Baseline PyTorch Models
- `thermal_baseline_cnn.pth` - PyTorch baseline CNN (85.11%)
- `thermal_augmented_cnn.pth` - Augmented CNN (29.78% - failed experiment)

Note: Model files are generated when running the notebook and have been cleaned for fresh training.

## Purpose

These experimental models were developed to:
1. Compare PyTorch vs TensorFlow implementations
2. Test ensemble methods for improved accuracy and reliability
3. Experiment with transfer learning approaches (ResNet-50)
4. Validate augmentation hypothesis across frameworks

## Latest Results (October 11, 2025)

| Model | Validation Accuracy | F1 (Macro) | Parameters | Notes |
|-------|---------------------|------------|------------|-------|
| **Ensemble (5 CNNs)** | **86.52%** | 0.8593 | 21.5M total | Best PyTorch result |
| **Baseline CNN (PyTorch)** | 85.11% | 0.8378 | 4.3M | Strong baseline |
| **TensorFlow Baseline** | 85.92% | 0.8522 | 3.3M | Production model |
| **Transfer ResNet-50 v2** | 84.91% | N/A | 24.6M | Thermal-adapted conv1 |
| **Transfer ResNet-50 v1** | 67.00% | N/A | 24.6M | ImageNet features, overfitting |
| **Augmented CNN (PyTorch)** | 29.78% | 0.2521 | 4.3M | Geometric aug failed (-55%) |

## Results

| Model Type | Parameters | Status |
|------------|------------|--------|
| **ResNet-152** | 65.7M | Experimental |
| **ResNet-50 Transfer** | ~25M | Testing |
| **Ensemble (5 models)** | Variable | Testing |
| **Baseline PyTorch** | ~87K | Comparison |

## Note

**Production Model**: The main production model is in the root directory:
- `thermal_emotion_baseline_model.h5` (TensorFlow/Keras)
- **85.92% validation accuracy**
- **3.3M parameters**
- **Production-ready**
- **Key insight**: Trained WITHOUT geometric augmentation - palette diversity alone provides superior results

**PyTorch Best Model**: Ensemble of 5 CNNs (experimental folder):
- `thermal_ensemble_model_1.pth` through `_5.pth`
- **86.52% ensemble accuracy** (best PyTorch result)
- **21.5M total parameters**
- **Research use**: More complex, requires loading 5 models

**Important Finding**: Both TensorFlow and PyTorch experiments confirm geometric augmentation (rotation, shift, zoom, flip) **dramatically decreases** performance:
- TensorFlow: 85.92% → 40.64% (-45.28%)
- PyTorch: 85.11% → 29.78% (-55.33%)

The multi-palette dataset structure provides natural color-based augmentation that is far more effective than traditional spatial transforms for thermal emotion recognition.

**Recommendation**: For production, use the TensorFlow baseline model (single file, lightweight, proven). For research into ensemble methods, PyTorch models demonstrate 1.4% improvement through model averaging.

## Documentation

For detailed experimental results and comparisons with the production baseline model, see:
- [`../documentation/PROJECT_DOCUMENTATION.md`](../documentation/PROJECT_DOCUMENTATION.md)

**Latest Results** (October 11, 2025):
- **TensorFlow Baseline**: 85.92% accuracy (production model)
- **PyTorch Ensemble**: 86.52% accuracy (5 models, best experimental result)
- **PyTorch Baseline**: 85.11% accuracy
- **PyTorch Transfer Learning v2**: 84.91% (thermal-adapted ResNet-50)
- **PyTorch Transfer Learning v1**: 67.00% (ImageNet features, severe overfitting)
- **Augmented Models**: Failed in both frameworks (-45-55% decrease)
- **Key Finding**: Palette diversity provides superior augmentation vs geometric transforms for thermal images

## Usage

### Setup PyTorch Environment (Python 3.11.6 Required)

```bash
# 1. Create Python 3.11.6 virtual environment
python3.11 -m venv .venv_pytorch

# 2. Activate the environment
.venv_pytorch\Scripts\activate  # Windows
# or
source .venv_pytorch/bin/activate  # Linux/Mac

# 3. Install PyTorch 2.8.0 with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install other dependencies
pip install opencv-python numpy pandas matplotlib scikit-learn jupyter

# 5. Verify GPU is available
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
```

### Expected Output (GPU Setup)
```
PyTorch version: 2.8.0+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

### Run Experimental Notebook

```bash
# Activate PyTorch environment
.venv_pytorch\Scripts\activate

# Launch Jupyter
jupyter notebook thermal_emotion_pytorch.ipynb

# The notebook will automatically use GPU for training
```

### Load a PyTorch Model (Example)

```python
import torch

# Load model
model = torch.load('thermal_resnet152_best.pth')
model.eval()

# Verify GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Model running on: {device}")
```

## Hardware Requirements

### GPU Training (Recommended)
- **GPU**: NVIDIA RTX 4060 or equivalent (8GB+ VRAM)
- **CUDA**: Version 11.8 or compatible
- **Python**: 3.11 (required for GPU compatibility)
- **RAM**: 16GB+ system memory
- **Training Time**: 
  - ResNet-152: ~6 hours for 40 epochs (~8-9 min/epoch)
  - Baseline CNN: ~15 minutes for 50 epochs

### CPU Training (Fallback)
- If GPU unavailable, PyTorch will use CPU
- Training times will be 10-20x slower
- Still requires Python 3.11 for compatibility

## Troubleshooting

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Python Version Issues
```bash
# Check Python version
python --version

# Should output: Python 3.11.6
# If not, create new environment with Python 3.11.6
python3.11 -m venv .venv_pytorch
```

### CUDA Out of Memory
- Reduce batch size in training
- Use smaller models (baseline CNN instead of ResNet-152)
- Close other GPU-intensive applications

---

**Recommendation**: Use these models for research and experimentation only. For production deployments, use the TensorFlow model in the root directory.
