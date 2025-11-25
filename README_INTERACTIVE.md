# VITON-HD Interactive Editor

An interactive web application built on top of VITON-HD that provides user-controllable editing features through post-processing and mask manipulation - **zero retraining required**.

## 🎯 Features

### 1️⃣ Color & Texture Editing
- **HSV Color Adjustment**: Modify hue (-180° to 180°), saturation (-100% to 100%), and brightness (-50% to 50%)
- **Real-time Preview**: See color changes instantly on the warped clothing
- Works entirely through post-processing of warped cloth output

### 2️⃣ Fit & Style Manipulation
- **Fit Presets**: Slim (-10%), Fitted (-5%), Relaxed (+5%), Oversized (+10%)
- **Sleeve Length Adjustment**: Extend or shorten sleeves (-20 to +20 pixels)
- **Mask Morphology**: Uses dilation/erosion for fit adjustments
- All changes applied through segmentation mask editing

### 3️⃣ Two-Tier Rendering
- **Quick Preview (256x192)**: Fast preview in ~5-10 seconds
- **HD Final Render (1024x768)**: High-quality output in ~60-90 seconds
- Smart caching to avoid redundant GMM computations

### 4️⃣ Interactive Web UI
- Built with **Streamlit** for easy interaction
- Dropdown selection from existing test dataset
- Real-time parameter adjustment with sliders
- Download buttons for preview and HD outputs
- Reset buttons for individual categories or all parameters

## 📁 Project Structure

```
VITON-HD/
├── app.py                    # Main Streamlit application
├── inference_pipeline.py     # Modular VITON-HD inference wrapper
├── editing_tools.py          # Color/texture and mask morphing tools
├── preprocessing.py          # Image loading and preprocessing utilities
├── test.py                   # Original test script (CPU-compatible)
├── networks.py               # Neural network architectures
├── datasets.py               # Dataset loaders
├── utils.py                  # Utility functions
├── test_pipeline.py          # Unit tests for pipeline
├── test_editing.py           # Unit tests for editing tools
├── checkpoints/              # Pretrained model weights
│   ├── seg_final.pth
│   ├── gmm_final.pth
│   └── alias_final.pth
└── datasets/                 # Test dataset
    └── test/
        ├── image/            # Person images (6 samples)
        ├── cloth/            # Clothing items (12 samples)
        ├── cloth-mask/       # Clothing masks
        ├── image-parse/      # Segmentation maps
        ├── openpose-img/     # Pose visualizations
        └── openpose-json/    # Pose keypoints
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8
- Conda environment with:
  - PyTorch 2.4.1 (CPU)
  - torchvision 0.20.0
  - opencv-python 4.12.0
  - torchgeometry 0.1.2
  - streamlit
  - PIL/Pillow

### Installation

1. **Activate the environment:**
```bash
conda activate viton-env
```

2. **Install additional dependencies:**
```bash
pip install streamlit
```

3. **Ensure datasets and checkpoints are in place:**
```
checkpoints/
├── alias_final.pth (384MB)
├── gmm_final.pth (73MB)
└── seg_final.pth (132MB)

datasets/test/
├── image/ (6 person images)
├── cloth/ (12 clothing items)
└── ... (other required files)
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

1. **Select Images**:
   - Choose a person from the dropdown (6 available)
   - Choose a clothing item from the dropdown (12 available)

2. **Adjust Colors** (optional):
   - Move the Hue slider to change color
   - Adjust Saturation for color intensity
   - Modify Brightness for lightness

3. **Adjust Fit** (optional):
   - Select a fit preset from dropdown
   - Fine-tune sleeve length with slider

4. **Generate Preview**:
   - Click "⚡ Quick Preview" for fast 256x192 preview (~5-10s)
   - Review the result in the middle column

5. **Generate HD**:
   - Click "🎯 Generate HD" for final 1024x768 output (~60-90s)
   - Download the result using the download button

6. **Reset**:
   - Use "🔄 Reset Color" or "🔄 Reset Fit" for specific resets
   - Use "🔄 Reset All" to return all parameters to defaults

## 🔧 Technical Details

### Architecture

**Pipeline Flow:**
```
Input (Person + Cloth)
    ↓
Preprocessing (load images, poses, masks)
    ↓
Segmentation Generator → Parse Map
    ↓
GMM (Geometric Matching) → Warped Cloth + Mask
    ↓
[EDITING LAYER - Color/Fit Adjustments]
    ↓
ALIAS Generator → Final Try-On Image
```

**Key Design Principles:**
- ✅ **Zero Retraining**: All models used in inference-only mode
- ✅ **Modular Design**: Inference pipeline exposed as reusable class
- ✅ **Efficient Caching**: Warped outputs cached to avoid redundant computation
- ✅ **CPU Compatible**: Runs on 16GB RAM CPU (no GPU required)

### Editing Mechanisms

**Color Editing:**
- Convert tensor from [-1, 1] to [0, 1]
- Apply OpenCV HSV transformations
- Convert back to tensor format
- Feed to ALIAS generator

**Fit Adjustment:**
- Apply morphological operations (dilate/erode) on warped cloth mask
- Adjust using OpenCV with elliptical kernels
- Recompute misalignment mask
- Feed adjusted mask to ALIAS generator

**Sleeve Length:**
- Shift sleeve regions in parse map vertically
- Use padding to extend/shorten
- Maintain spatial consistency

## 🎨 Examples

### Color Transformations
- **Red → Blue**: Hue shift +120°
- **Vibrant**: Saturation +50%
- **Darker**: Brightness -30%

### Fit Variations
- **Slim Fit**: -10% (tighter around body)
- **Oversized**: +10% (looser fit)
- **Short Sleeves**: -15 pixels
- **Long Sleeves**: +15 pixels

## ⚡ Performance

**Hardware:** 16GB RAM CPU (Intel i7)

**Inference Times:**
- Quick Preview (256x192): ~5-10 seconds
- HD Render (1024x768): ~60-90 seconds
- Color/Fit Edits: Instant (no model inference)

**Memory Usage:**
- Peak: ~3-4 GB
- Models: ~589 MB (checkpoints)
- Single Batch: ~500 MB

## 🧪 Testing

Run unit tests to verify functionality:

```bash
# Test inference pipeline
python test_pipeline.py

# Test editing tools
python test_editing.py
```

## 📝 Limitations

1. **CPU Performance**: Inference is slower than GPU (~60x)
   - Mitigated with two-tier preview system

2. **Mask Quality**: Aggressive fit adjustments may create artifacts
   - Conservative default ranges to prevent unrealistic results

3. **Input Requirements**: Requires pre-processed test data
   - Pose keypoints and segmentation maps must exist
   - Not automatic end-to-end from raw images

4. **Pattern Realism**: Simple overlays don't respect fabric physics
   - Works best for basic color/texture changes

## 🛠️ Future Enhancements

- [ ] GPU acceleration support
- [ ] File upload for custom images (with auto pose/segmentation)
- [ ] Pattern library with pre-made textures
- [ ] Batch processing multiple pairs
- [ ] Side-by-side comparison view
- [ ] Export editing presets as JSON
- [ ] Video generation from parameter animations

## 📄 License

This project builds upon VITON-HD:
- Original VITON-HD: Creative Commons BY-NC 4.0
- Interactive Editor Extensions: Same license

## 🙏 Acknowledgments

- **VITON-HD Team**: Original paper and pretrained models
- **Shadow2496**: GitHub repository
- Built for educational purposes (EE782 project)

## 📧 Contact

For issues, questions, or contributions, please open an issue on GitHub.

---

**Note**: This is an MVP (Minimum Viable Product) demonstrating zero-retraining editing capabilities. The system successfully proves that interactive virtual try-on editing is possible through pure post-processing without model retraining.
