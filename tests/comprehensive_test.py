"""
Comprehensive test script for VITON-HD editing capabilities.
Tests various color, brightness, fit, and sleeve adjustments.
"""

import os
import torch
from PIL import Image
import numpy as np

from src.inference_pipeline import VITONInference
from src.preprocessing import VITONPreprocessor
from src.cloth_editor import ColorEditor, MaskMorpher

def tensor_to_image(tensor):
    """Convert tensor to PIL Image."""
    img = tensor.squeeze(0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def save_result(output_tensor, save_path, label):
    """Save output tensor as image."""
    img = tensor_to_image(output_tensor)
    img.save(save_path)
    print(f"✓ Saved {label}: {save_path}")

def main():
    print("=" * 80)
    print("VITON-HD Comprehensive Editing Test")
    print("=" * 80)
    
    # Setup paths
    dataset_dir = './datasets'  # Base dataset directory
    output_dir = './results/comprehensive_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Test configuration
    person_name = '00891_00.jpg'  # Person image name
    cloth_name = '01260_00.jpg'   # Cloth image name
    
    print(f"\nTest Configuration:")
    print(f"  Person: {person_name}")
    print(f"  Cloth: {cloth_name}")
    print(f"  Output: {output_dir}")
    
    # Initialize components
    print("\n" + "-" * 80)
    print("Initializing VITON-HD Pipeline...")
    print("-" * 80)
    
    pipeline = VITONInference()
    pipeline.load_models(
        seg_checkpoint='seg_final.pth',
        gmm_checkpoint='gmm_final.pth',
        alias_checkpoint='alias_final.pth'
    )
    pipeline.set_resolution(width=768, height=1024)  # Full HD resolution
    
    preprocessor = VITONPreprocessor(dataset_dir)
    color_editor = ColorEditor()
    mask_morpher = MaskMorpher()
    
    # Load data
    print("\nLoading person and cloth data...")
    person_data = preprocessor.load_person_data(person_name)
    cloth_data = preprocessor.load_cloth_data(cloth_name)
    
    # Extract tensors
    img_agnostic = person_data['img_agnostic']
    parse_agnostic = person_data['parse_agnostic']
    pose = person_data['pose']
    c = cloth_data['cloth']
    cm = cloth_data['cloth_mask']
    
    print("✓ Data loaded successfully")
    
    # Test 1: Baseline (no editing)
    print("\n" + "-" * 80)
    print("Test 1: Baseline (No Editing)")
    print("-" * 80)
    
    output_baseline, intermediates = pipeline.run_full_pipeline(
        img_agnostic, parse_agnostic, pose, c, cm,
        return_intermediates=True
    )
    save_result(output_baseline, f"{output_dir}/01_baseline.jpg", "Baseline")
    
    # Extract intermediates for editing
    warped_c = intermediates['warped_c']
    warped_cm = intermediates['warped_cm']
    parse = intermediates['parse']
    
    # Test 2: Color Adjustments
    print("\n" + "-" * 80)
    print("Test 2: Color Adjustments")
    print("-" * 80)
    
    color_tests = [
        {"hue": 30, "sat": 0, "bright": 0, "label": "Hue +30 (Red shift)"},
        {"hue": -30, "sat": 0, "bright": 0, "label": "Hue -30 (Blue shift)"},
        {"hue": 0, "sat": 50, "bright": 0, "label": "Saturation +50 (More vibrant)"},
        {"hue": 0, "sat": -50, "bright": 0, "label": "Saturation -50 (Desaturated)"},
        {"hue": 0, "sat": 0, "bright": 30, "label": "Brightness +30 (Lighter)"},
        {"hue": 0, "sat": 0, "bright": -30, "label": "Brightness -30 (Darker)"},
        {"hue": 20, "sat": 30, "bright": 10, "label": "Combined (Hue+20, Sat+30, Bright+10)"},
    ]
    
    for i, test in enumerate(color_tests, start=2):
        print(f"  {test['label']}...", end=" ")
        warped_c_edited = color_editor.adjust_hsv(
            warped_c.clone(),
            hue_shift=test['hue'],
            saturation_shift=test['sat'],
            brightness_shift=test['bright']
        )
        
        output = pipeline.run_alias(img_agnostic, pose, warped_c_edited, parse, warped_cm)
        save_path = f"{output_dir}/{i:02d}_color_{test['hue']}h_{test['sat']}s_{test['bright']}b.jpg"
        save_result(output, save_path, test['label'])
    
    # Test 3: Fit Adjustments
    print("\n" + "-" * 80)
    print("Test 3: Fit Adjustments")
    print("-" * 80)
    
    fit_tests = [
        {"percent": 25, "label": "Fit +25% (Looser)"},
        {"percent": 50, "label": "Fit +50% (Much looser)"},
        {"percent": -15, "label": "Fit -15% (Tighter)"},
        {"percent": -25, "label": "Fit -25% (Much tighter)"},
    ]
    
    base_idx = len(color_tests) + 2
    for i, test in enumerate(fit_tests, start=base_idx):
        print(f"  {test['label']}...", end=" ")
        warped_cm_edited = mask_morpher.adjust_fit(
            warped_cm.clone(),
            fit_percentage=test['percent']
        )
        
        output = pipeline.run_alias(img_agnostic, pose, warped_c, parse, warped_cm_edited)
        save_path = f"{output_dir}/{i:02d}_fit_{test['percent']:+d}pct.jpg"
        save_result(output, save_path, test['label'])
    
    # Test 4: Sleeve Length Adjustments
    # Test 4: Sleeve Length Adjustments (DISABLED - causes artifacts)
    print("\n" + "-" * 80)
    print("Test 4: Sleeve Length Adjustments (SKIPPED - causes artifacts)")
    print("-" * 80)
    print("  Sleeve adjustments disabled due to visual artifacts")
    
    sleeve_tests = []  # Disabled for now
    
    # Test 5: Fit Presets
    print("\n" + "-" * 80)
    print("Test 5: Fit Presets")
    print("-" * 80)
    
    preset_tests = [
        {"preset": "slim", "label": "Preset: Slim Fit"},
        {"preset": "regular", "label": "Preset: Regular Fit"},
        {"preset": "relaxed", "label": "Preset: Relaxed Fit"},
        {"preset": "oversized", "label": "Preset: Oversized Fit"},
    ]
    
    base_idx = len(color_tests) + len(fit_tests) + len(sleeve_tests) + 2
    for i, test in enumerate(preset_tests, start=base_idx):
        print(f"  {test['label']}...", end=" ")
        warped_cm_edited = mask_morpher.apply_fit_preset(
            warped_cm.clone(),
            preset_name=test['preset']
        )
        
        output = pipeline.run_alias(img_agnostic, pose, warped_c, parse, warped_cm_edited)
        save_path = f"{output_dir}/{i:02d}_preset_{test['preset']}.jpg"
        save_result(output, save_path, test['label'])
    
    # Test 6: Combined Edits
    print("\n" + "-" * 80)
    print("Test 6: Combined Edits")
    print("-" * 80)
    
    combined_tests = [
        {
            "hue": 15, "sat": 20, "bright": 10,
            "fit": 30,
            "label": "Combined: Warmer color + Looser fit"
        },
        {
            "hue": -20, "sat": 30, "bright": 0,
            "fit": -20,
            "label": "Combined: Cooler vibrant + Tighter fit"
        },
        {
            "hue": 0, "sat": -30, "bright": 20,
            "fit": 40,
            "label": "Combined: Lighter muted + Oversized fit"
        },
    ]
    
    base_idx = len(color_tests) + len(fit_tests) + len(sleeve_tests) + len(preset_tests) + 2
    for i, test in enumerate(combined_tests, start=base_idx):
        print(f"  {test['label']}...", end=" ")
        
        # Apply color edits
        warped_c_edited = color_editor.adjust_hsv(
            warped_c.clone(),
            hue_shift=test['hue'],
            saturation_shift=test['sat'],
            brightness_shift=test['bright']
        )
        
        # Apply fit edits
        warped_cm_edited = mask_morpher.adjust_fit(
            warped_cm.clone(),
            fit_percentage=test['fit']
        )
        
        output = pipeline.run_alias(img_agnostic, pose, warped_c_edited, parse, warped_cm_edited)
        save_path = f"{output_dir}/{i:02d}_combined_{i-base_idx+1}.jpg"
        save_result(output, save_path, test['label'])
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    total_tests = (1 + len(color_tests) + len(fit_tests) + len(sleeve_tests) + 
                   len(preset_tests) + len(combined_tests))
    print(f"✓ Total tests completed: {total_tests}")
    print(f"  - Baseline: 1")
    print(f"  - Color adjustments: {len(color_tests)}")
    print(f"  - Fit adjustments: {len(fit_tests)}")
    print(f"  - Sleeve adjustments: {len(sleeve_tests)}")
    print(f"  - Fit presets: {len(preset_tests)}")
    print(f"  - Combined edits: {len(combined_tests)}")
    print(f"\nAll results saved to: {output_dir}/")
    print("=" * 80)

if __name__ == "__main__":
    main()
