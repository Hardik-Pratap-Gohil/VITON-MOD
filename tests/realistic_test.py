"""
Realistic editing test for VITON-HD.
Tests colors, patterns, logos, and textures on source cloth.
"""

import os
import torch
from PIL import Image
import numpy as np

from src.inference_pipeline import VITONInference
from src.preprocessing import VITONPreprocessor
from src.cloth_editor import ClothColorEditor, PatternGenerator, LogoApplicator, FabricSimulator

def tensor_to_image(tensor):
    """Convert tensor to PIL Image."""
    img = tensor.squeeze(0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img + 1) / 2.0
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def save_result(output_tensor, save_path, label):
    """Save output tensor as image."""
    img = tensor_to_image(output_tensor)
    img.save(save_path)
    print(f"✓ Saved {label}: {save_path}")

def main():
    print("=" * 80)
    print("VITON-HD Realistic Editing Test")
    print("=" * 80)
    
    # Setup
    dataset_dir = './datasets'
    output_dir = './results/realistic_test'
    os.makedirs(output_dir, exist_ok=True)
    
    person_name = '00891_00.jpg'
    cloth_name = '01260_00.jpg'
    
    print(f"\nTest Configuration:")
    print(f"  Person: {person_name}")
    print(f"  Cloth: {cloth_name}")
    print(f"  Output: {output_dir}")
    
    # Initialize
    print("\n" + "-" * 80)
    print("Initializing Pipeline...")
    print("-" * 80)
    
    pipeline = VITONInference()
    pipeline.load_models(
        seg_checkpoint='seg_final.pth',
        gmm_checkpoint='gmm_final.pth',
        alias_checkpoint='alias_final.pth'
    )
    pipeline.set_resolution(width=768, height=1024)
    
    preprocessor = VITONPreprocessor(dataset_dir)
    
    # Load data
    print("\nLoading data...")
    person_data = preprocessor.load_person_data(person_name)
    cloth_data = preprocessor.load_cloth_data(cloth_name)
    
    img_agnostic = person_data['img_agnostic']
    parse_agnostic = person_data['parse_agnostic']
    pose = person_data['pose']
    c = cloth_data['cloth']
    cm = cloth_data['cloth_mask']
    
    print("✓ Data loaded")
    
    # Test 1: Baseline
    print("\n" + "-" * 80)
    print("Test 1: Baseline (Original Cloth)")
    print("-" * 80)
    
    output_baseline, _ = pipeline.run_full_pipeline(
        img_agnostic, parse_agnostic, pose, c, cm,
        return_intermediates=True
    )
    save_result(output_baseline, f"{output_dir}/01_baseline.jpg", "Baseline")
    
    # Test 2: Color Palettes
    print("\n" + "-" * 80)
    print("Test 2: Color Palettes")
    print("-" * 80)
    
    palettes = ['vibrant', 'pastel', 'earth', 'monochrome', 'warm', 'cool']
    
    for i, palette in enumerate(palettes, start=2):
        print(f"  Palette: {palette}...", end=" ")
        c_colored = ClothColorEditor.apply_color_palette(c.clone(), cm, palette)
        output, _ = pipeline.run_full_pipeline(
            img_agnostic, parse_agnostic, pose, c_colored, cm,
            return_intermediates=True
        )
        save_result(output, f"{output_dir}/{i:02d}_palette_{palette}.jpg", palette)
    
    # Test 3: Custom Color Adjustments
    print("\n" + "-" * 80)
    print("Test 3: Custom Colors")
    print("-" * 80)
    
    color_tests = [
        {"hue": 30, "sat": 0, "bright": 0, "label": "Red shift"},
        {"hue": -30, "sat": 0, "bright": 0, "label": "Blue shift"},
        {"hue": 0, "sat": 50, "bright": 0, "label": "More saturated"},
        {"hue": 0, "sat": 0, "bright": 30, "label": "Brighter"},
    ]
    
    base_idx = 2 + len(palettes)
    for i, test in enumerate(color_tests, start=base_idx):
        print(f"  {test['label']}...", end=" ")
        c_adjusted = ClothColorEditor.adjust_hsv(
            c.clone(), cm,
            hue_shift=test['hue'],
            saturation_shift=test['sat'],
            brightness_shift=test['bright']
        )
        output, _ = pipeline.run_full_pipeline(
            img_agnostic, parse_agnostic, pose, c_adjusted, cm,
            return_intermediates=True
        )
        save_result(output, f"{output_dir}/{i:02d}_color_{test['label'].replace(' ', '_').lower()}.jpg", test['label'])
    
    # Test 4: Patterns
    print("\n" + "-" * 80)
    print("Test 4: Patterns")
    print("-" * 80)
    
    # Get cloth dimensions
    _, _, h, w = c.shape
    
    pattern_tests = [
        {
            "name": "vertical_stripes",
            "pattern": PatternGenerator.create_stripes(h, w, 'vertical', 15, (100, 100, 255), (255, 255, 255)),
            "blend": "multiply",
            "opacity": 0.6
        },
        {
            "name": "horizontal_stripes",
            "pattern": PatternGenerator.create_stripes(h, w, 'horizontal', 20, (255, 100, 100), (255, 255, 255)),
            "blend": "multiply",
            "opacity": 0.6
        },
        {
            "name": "polkadots",
            "pattern": PatternGenerator.create_polkadots(h, w, 8, 35, (255, 255, 255), (50, 50, 50)),
            "blend": "multiply",
            "opacity": 0.5
        },
        {
            "name": "checkerboard",
            "pattern": PatternGenerator.create_checkerboard(h, w, 25, (200, 200, 200), (100, 100, 100)),
            "blend": "overlay",
            "opacity": 0.4
        },
    ]
    
    base_idx = base_idx + len(color_tests)
    for i, test in enumerate(pattern_tests, start=base_idx):
        print(f"  Pattern: {test['name']}...", end=" ")
        c_patterned = PatternGenerator.apply_pattern(
            c.clone(), cm, test['pattern'], test['blend'], test['opacity']
        )
        output, _ = pipeline.run_full_pipeline(
            img_agnostic, parse_agnostic, pose, c_patterned, cm,
            return_intermediates=True
        )
        save_result(output, f"{output_dir}/{i:02d}_pattern_{test['name']}.jpg", test['name'])
    
    # Test 5: Text Logos
    print("\n" + "-" * 80)
    print("Test 5: Text Logos")
    print("-" * 80)
    
    logo_tests = [
        {"text": "BRAND", "pos": "center", "size": 60, "color": (255, 255, 255)},
        {"text": "STYLE", "pos": "top", "size": 50, "color": (0, 0, 0)},
        {"text": "2025", "pos": "bottom", "size": 40, "color": (200, 200, 200)},
    ]
    
    base_idx = base_idx + len(pattern_tests)
    for i, test in enumerate(logo_tests, start=base_idx):
        print(f"  Logo: {test['text']} ({test['pos']})...", end=" ")
        c_logo = LogoApplicator.add_text_logo(
            c.clone(), cm, test['text'], test['pos'], test['size'], test['color']
        )
        output, _ = pipeline.run_full_pipeline(
            img_agnostic, parse_agnostic, pose, c_logo, cm,
            return_intermediates=True
        )
        save_result(output, f"{output_dir}/{i:02d}_logo_{test['text'].lower()}_{test['pos']}.jpg", 
                   f"{test['text']} {test['pos']}")
    
    # Test 6: Fabric Textures
    print("\n" + "-" * 80)
    print("Test 6: Fabric Textures")
    print("-" * 80)
    
    textures = ['canvas', 'denim', 'silk', 'linen']
    
    base_idx = base_idx + len(logo_tests)
    for i, texture in enumerate(textures, start=base_idx):
        print(f"  Texture: {texture}...", end=" ")
        c_textured = FabricSimulator.add_subtle_texture(c.clone(), cm, texture)
        output, _ = pipeline.run_full_pipeline(
            img_agnostic, parse_agnostic, pose, c_textured, cm,
            return_intermediates=True
        )
        save_result(output, f"{output_dir}/{i:02d}_texture_{texture}.jpg", texture)
    
    # Test 7: Combined Effects
    print("\n" + "-" * 80)
    print("Test 7: Combined Effects")
    print("-" * 80)
    
    combined_tests = [
        {
            "label": "Vibrant stripes + logo",
            "ops": [
                lambda x, m: ClothColorEditor.apply_color_palette(x, m, 'vibrant'),
                lambda x, m: PatternGenerator.apply_pattern(
                    x, m, PatternGenerator.create_stripes(h, w, 'vertical', 12, (255, 200, 200), (255, 255, 255)),
                    'multiply', 0.5
                ),
                lambda x, m: LogoApplicator.add_text_logo(x, m, "SPORT", "center", 55, (50, 50, 50))
            ]
        },
        {
            "label": "Pastel polkadots",
            "ops": [
                lambda x, m: ClothColorEditor.apply_color_palette(x, m, 'pastel'),
                lambda x, m: PatternGenerator.apply_pattern(
                    x, m, PatternGenerator.create_polkadots(h, w, 6, 30, (255, 255, 255), (150, 150, 200)),
                    'overlay', 0.4
                )
            ]
        },
        {
            "label": "Denim texture + brand",
            "ops": [
                lambda x, m: FabricSimulator.add_subtle_texture(x, m, 'denim'),
                lambda x, m: ClothColorEditor.adjust_hsv(x, m, -15, 10, -5),
                lambda x, m: LogoApplicator.add_text_logo(x, m, "DENIM", "top", 45, (255, 255, 255))
            ]
        },
    ]
    
    base_idx = base_idx + len(textures)
    for i, test in enumerate(combined_tests, start=base_idx):
        print(f"  {test['label']}...", end=" ")
        c_combined = c.clone()
        for op in test['ops']:
            c_combined = op(c_combined, cm)
        
        output, _ = pipeline.run_full_pipeline(
            img_agnostic, parse_agnostic, pose, c_combined, cm,
            return_intermediates=True
        )
        save_result(output, f"{output_dir}/{i:02d}_combined_{i-base_idx+1}.jpg", test['label'])
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    total_tests = (1 + len(palettes) + len(color_tests) + len(pattern_tests) + 
                   len(logo_tests) + len(textures) + len(combined_tests))
    print(f"✓ Total tests completed: {total_tests}")
    print(f"  - Baseline: 1")
    print(f"  - Color palettes: {len(palettes)}")
    print(f"  - Custom colors: {len(color_tests)}")
    print(f"  - Patterns: {len(pattern_tests)}")
    print(f"  - Text logos: {len(logo_tests)}")
    print(f"  - Fabric textures: {len(textures)}")
    print(f"  - Combined effects: {len(combined_tests)}")
    print(f"\nAll results saved to: {output_dir}/")
    print("=" * 80)

if __name__ == "__main__":
    main()
