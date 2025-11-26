# VITON-MOD: Interactive Cloth Editor for VITON-HD

An interactive web application built on top of [VITON-HD](https://arxiv.org/abs/2103.16874) (Choi et al., CVPR 2021) that enables realistic cloth editing through colors, patterns, logos, and textures **without retraining any models**. 

## ✨ Features

- **🎨 Color Editing**

  - 6 preset palettes (Vibrant, Pastel, Earth, Monochrome, Warm, Cool)

  - Custom HSV adjustments (Hue, Saturation, Brightness)

- **🎭 Pattern Overlay**

  - Procedurally generated patterns (Stripes, Polkadots, Checkerboard)

  - Customizable colors, sizes, and blend modes

  - Preserves original cloth lighting and shadows

- **🏷️ Logo Placement**

  - Add text logos with custom text, position, size, and color

  - Automatically warps with the cloth for realistic appearance

- **🧵 Fabric Textures**

  - Simulate different materials (Canvas, Denim, Silk, Linen)

  - Subtle texture enhancements without artifacts
  - 

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 16GB RAM (CPU-only, no GPU required)
- conda (recommended for environment management)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Hardik-Pratap-Gohil/VITON-MOD.git
cd VITON-MOD
```

2. **Create conda environment**
```bash
conda create -n viton-env python=3.8
conda activate viton-env
```

3. **Install dependencies**
```bash
pip install torch==2.4.1 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python==4.12.0.90 pillow numpy streamlit
pip install torchgeometry
```

4. **Download pretrained models**

Place the following checkpoints in `./checkpoints/`:
- `seg_final.pth` (132 MB) - Segmentation model
- `gmm_final.pth` (73 MB) - Geometric Matching Module
- `alias_final.pth` (384 MB) - ALIAS Generator

5. **Prepare dataset** (optional - sample dataset included)

Place test images in `./datasets/test/`:
```
datasets/test/
├── image/              # Person images
├── cloth/              # Cloth images  
├── cloth-mask/         # Cloth masks
├── image-parse/        # Segmentation maps
├── openpose-img/       # Pose visualizations
└── openpose-json/      # Pose keypoints
```

### Running the App

**Start the Streamlit web application:**
```bash
# Using the convenience script
./scripts/run_app.sh

# Or directly
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

1. **Select Person & Cloth**: Choose from dropdown menus in the sidebar
2. **Choose Resolution**: Preview (fast) or HD (high quality)
3. **Edit Cloth**: Use tabs to apply colors, patterns, logos, or textures
4. **Generate**: Click "Generate Try-On" to see results
5. **Download**: Save the final image using the download button

## 🧪 Testing

Run comprehensive tests to verify all editing capabilities:

```bash
# Using convenience script
./scripts/run_tests.sh

# Or run tests individually
python tests/realistic_test.py  # Test colors, patterns, logos, textures (24 images)
python tests/logo_test.py       # Test logo placement variations (25 images)
python tests/test.py            # Run original VITON-HD inference
```

Results are saved in `./results/` with organized subdirectories.

## 📁 Repository Structure

```
VITON-MOD/
├── app.py                      # Streamlit web application
│
├── src/                        # Core source code
│   ├── __init__.py
│   ├── cloth_editor.py         # Editing tools (colors, patterns, logos, textures)
│   ├── inference_pipeline.py   # VITON-HD pipeline wrapper
│   ├── preprocessing.py        # Data loading utilities
│   ├── networks.py             # Neural network architectures
│   ├── utils.py                # Helper functions
│   └── datasets.py             # Dataset class (original VITON-HD)
│
├── tests/                      # Test scripts
│   ├── __init__.py
│   ├── realistic_test.py       # Comprehensive editing tests (24 images)
│   ├── logo_test.py            # Logo placement tests (25 images)
│   ├── comprehensive_test.py   # General feature tests
│   └── test.py                 # Original VITON-HD inference
│
├── scripts/                    # Utility scripts
│   ├── run_app.sh              # Launch Streamlit app
│   └── run_tests.sh            # Run all tests
│
├── checkpoints/                # Pretrained model weights
│   ├── seg_final.pth           # Segmentation model (132 MB)
│   ├── gmm_final.pth           # Geometric Matching Module (73 MB)
│   └── alias_final.pth         # ALIAS Generator (384 MB)
│
├── datasets/                   # Test data
│   ├── test_pairs.txt
│   └── test/
│       ├── image/              # 6 person images
│       ├── cloth/              # 12 cloth items
│       ├── cloth-mask/         # Cloth segmentation masks
│       ├── image-parse/        # Person segmentation maps
│       ├── openpose-img/       # Pose visualizations
│       └── openpose-json/      # Pose keypoints (JSON)
│
├── assets/                     # Optional custom content
│   ├── logos/                  # Custom logo images (PNG)
│   ├── patterns/               # Custom pattern tiles
│   └── accessories/            # Reference images
│
└── results/                    # Generated outputs
    ├── realistic_test/         # Realistic editing test results
    ├── logo_test/              # Logo placement test results
    └── comprehensive_test/     # General test results
```

## 🎯 Design

### Why Edit the Source Cloth?

Unlike traditional approaches that modify intermediate pipeline outputs (which causes artifacts), VITON-MOD applies all edits to the **source cloth image** before it enters the VITON-HD pipeline. This ensures:

✅ **No artifacts** - VITON-HD's GMM naturally warps the edited cloth  
✅ **Preserved alignment** - Cloth and mask remain perfectly synchronized  
✅ **Realistic results** - Original lighting, shadows, and folds are maintained  

### What Doesn't Work (and why we don't do it)

❌ **Fit adjustments via mask morphology** - Breaks cloth-mask alignment, causes artifacts  
❌ **Sleeve length modifications** - Creates visible distortions  
❌ **Post-processing on warped outputs** - Disrupts carefully learned features  

## 🔧 Technical Details

- **Framework**: Built on VITON-HD (CVPR 2021) pretrained models
- **Inference**: CPU-only, ~60-90 seconds per 1024×768 image
- **Editing**: Pre-processing approach (edit source → warp naturally)
- **UI**: Streamlit for interactive web interface
- **No Training Required**: Pure post-processing and pre-processing techniques

## 📚 Citation

```bibtex
@inproceedings{choi2021viton,
  title={VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization},
  author={Choi, Seunghwan and Park, Sunghyun and Lee, Minsoo and Choo, Jaegul},
  booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

## 🔗 References

- [VITON-HD Paper](https://arxiv.org/abs/2103.16874)
- [Original VITON-HD Implementation](https://github.com/shadow2496/VITON-HD)
- [VITON-MOD Repository](https://github.com/Hardik-Pratap-Gohil/VITON-MOD)

## 🙏 Acknowledgments

Built upon the excellent work of Choi et al. in VITON-HD. This project extends their framework with interactive editing capabilities for educational and research purposes.
