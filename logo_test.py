"""
Logo placement test for VITON-HD.
Tests various text logos with different words, positions, sizes, and colors.
"""

import os
import torch
from PIL import Image
import numpy as np

from inference_pipeline import VITONInference
from preprocessing import VITONPreprocessor
from cloth_editor import LogoApplicator

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
    print("VITON-HD Logo Placement Test")
    print("=" * 80)
    
    # Setup
    dataset_dir = './datasets'
    output_dir = './results/logo_test'
    os.makedirs(output_dir, exist_ok=True)
    
    person_name = '00891_00.jpg'
    cloth_name = '01260_00.jpg'
    
    print(f"\nConfiguration:")
    print(f"  Person: {person_name}")
    print(f"  Cloth: {cloth_name}")
    print(f"  Output: {output_dir}")
    
    # Initialize
    print("\n" + "-" * 80)
    print("Initializing...")
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
    
    # Baseline
    print("\n" + "-" * 80)
    print("Generating logo variations...")
    print("-" * 80)
    
    logo_tests = [
        # Brand names - center
        {"text": "NIKE", "pos": "center", "size": 70, "color": (255, 255, 255), "label": "NIKE_center_white"},
        {"text": "ADIDAS", "pos": "center", "size": 65, "color": (0, 0, 0), "label": "ADIDAS_center_black"},
        {"text": "PUMA", "pos": "center", "size": 75, "color": (200, 200, 200), "label": "PUMA_center_gray"},
        {"text": "GUCCI", "pos": "center", "size": 80, "color": (180, 150, 100), "label": "GUCCI_center_gold"},
        
        # Sports/Athletic - center
        {"text": "SPORT", "pos": "center", "size": 60, "color": (255, 100, 100), "label": "SPORT_center_red"},
        {"text": "ATHLETICS", "pos": "center", "size": 55, "color": (100, 100, 255), "label": "ATHLETICS_center_blue"},
        {"text": "TEAM", "pos": "center", "size": 70, "color": (50, 50, 50), "label": "TEAM_center_dark"},
        
        # Numbers/Years - top
        {"text": "23", "pos": "top", "size": 90, "color": (255, 255, 255), "label": "23_top_white"},
        {"text": "2025", "pos": "top", "size": 70, "color": (220, 220, 220), "label": "2025_top_lightgray"},
        {"text": "99", "pos": "top", "size": 100, "color": (255, 200, 0), "label": "99_top_yellow"},
        
        # Short words - top
        {"text": "NYC", "pos": "top", "size": 75, "color": (0, 0, 0), "label": "NYC_top_black"},
        {"text": "USA", "pos": "top", "size": 70, "color": (200, 50, 50), "label": "USA_top_red"},
        {"text": "PRO", "pos": "top", "size": 65, "color": (255, 255, 255), "label": "PRO_top_white"},
        
        # Words - bottom
        {"text": "ORIGINAL", "pos": "bottom", "size": 50, "color": (100, 100, 100), "label": "ORIGINAL_bottom_gray"},
        {"text": "VINTAGE", "pos": "bottom", "size": 55, "color": (150, 100, 70), "label": "VINTAGE_bottom_brown"},
        {"text": "ELITE", "pos": "bottom", "size": 60, "color": (255, 255, 255), "label": "ELITE_bottom_white"},
        
        # Fashion brands - center, different sizes
        {"text": "FASHION", "pos": "center", "size": 50, "color": (0, 0, 0), "label": "FASHION_center_small"},
        {"text": "LUXURY", "pos": "center", "size": 85, "color": (180, 150, 120), "label": "LUXURY_center_large"},
        {"text": "STYLE", "pos": "center", "size": 65, "color": (255, 255, 255), "label": "STYLE_center_medium"},
        
        # Urban/Street - various
        {"text": "URBAN", "pos": "top", "size": 60, "color": (50, 50, 50), "label": "URBAN_top_dark"},
        {"text": "STREET", "pos": "center", "size": 70, "color": (255, 255, 255), "label": "STREET_center_white"},
        {"text": "SKATE", "pos": "bottom", "size": 65, "color": (200, 200, 200), "label": "SKATE_bottom_gray"},
        
        # Classic/Timeless
        {"text": "CLASSIC", "pos": "center", "size": 60, "color": (100, 100, 100), "label": "CLASSIC_center_gray"},
        {"text": "PARIS", "pos": "top", "size": 65, "color": (0, 0, 0), "label": "PARIS_top_black"},
        {"text": "TOKYO", "pos": "center", "size": 70, "color": (200, 50, 50), "label": "TOKYO_center_red"},
    ]
    
    for i, test in enumerate(logo_tests, start=1):
        print(f"  [{i}/{len(logo_tests)}] {test['label']}...", end=" ")
        
        c_logo = LogoApplicator.add_text_logo(
            c.clone(), cm, 
            text=test['text'], 
            position=test['pos'], 
            font_size=test['size'], 
            color=test['color']
        )
        
        output, _ = pipeline.run_full_pipeline(
            img_agnostic, parse_agnostic, pose, c_logo, cm,
            return_intermediates=True
        )
        
        save_result(output, f"{output_dir}/{i:02d}_{test['label']}.jpg", test['label'])
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)
    print(f"✓ Generated {len(logo_tests)} logo variations")
    print(f"  Output directory: {output_dir}/")
    print("=" * 80)

if __name__ == "__main__":
    main()
