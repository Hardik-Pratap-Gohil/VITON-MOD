"""
Test script for editing tools
"""

import torch
from editing_tools import ColorEditor, MaskMorpher
from inference_pipeline import VITONInference
from datasets import VITONDataset, VITONDataLoader
from utils import save_images
import os

class TestOpt:
    def __init__(self):
        self.load_height = 1024
        self.load_width = 768
        self.semantic_nc = 13
        self.dataset_dir = './datasets/'
        self.dataset_mode = 'test'
        self.dataset_list = 'test_pairs.txt'
        self.batch_size = 1
        self.workers = 1
        self.shuffle = False

def test_editing_tools():
    print("Testing Editing Tools...")
    
    # Initialize pipeline
    pipeline = VITONInference(checkpoint_dir='./checkpoints/')
    pipeline.load_models()
    
    # Load test data
    opt = TestOpt()
    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)
    inputs = next(iter(test_loader.data_loader))
    
    print(f"\nProcessing: {inputs['img_name'][0]} + {inputs['c_name']['unpaired'][0]}")
    
    # Run baseline
    print("\n1. Running baseline...")
    output_baseline, intermediates = pipeline.run_full_pipeline(
        inputs['img_agnostic'],
        inputs['parse_agnostic'],
        inputs['pose'],
        inputs['cloth']['unpaired'],
        inputs['cloth_mask']['unpaired'],
        return_intermediates=True
    )
    
    # Test color editing
    print("2. Testing color editing (Hue +60, Saturation +30)...")
    edited_warped_c = ColorEditor.adjust_hsv(
        intermediates['warped_c'],
        hue_shift=60,
        saturation_shift=30,
        brightness_shift=10
    )
    output_color = pipeline.run_with_edited_cloth(intermediates, edited_warped_c)
    
    # Test fit adjustment
    print("3. Testing fit adjustment (Looser +10%)...")
    edited_cm = MaskMorpher.adjust_fit(intermediates['warped_cm'], fit_percentage=10)
    output_fit = pipeline.run_alias(
        intermediates['img_agnostic'],
        intermediates['pose'],
        intermediates['warped_c'],
        intermediates['parse'],
        edited_cm
    )
    
    # Test preset
    print("4. Testing fit preset (Oversized)...")
    preset_cm = MaskMorpher.apply_fit_preset(intermediates['warped_cm'], 'Oversized (+10%)')
    output_preset = pipeline.run_alias(
        intermediates['img_agnostic'],
        intermediates['pose'],
        intermediates['warped_c'],
        intermediates['parse'],
        preset_cm
    )
    
    # Save results
    print("\n5. Saving test results...")
    os.makedirs('./results/editing_test/', exist_ok=True)
    
    save_images(output_baseline, ['baseline.jpg'], './results/editing_test/')
    save_images(output_color, ['color_edited.jpg'], './results/editing_test/')
    save_images(output_fit, ['fit_adjusted.jpg'], './results/editing_test/')
    save_images(output_preset, ['preset_oversized.jpg'], './results/editing_test/')
    
    print("\n✅ All editing tests passed!")
    print("   Results saved to ./results/editing_test/")
    print("   - baseline.jpg (original)")
    print("   - color_edited.jpg (hue +60, sat +30)")
    print("   - fit_adjusted.jpg (looser +10%)")
    print("   - preset_oversized.jpg (oversized preset)")

if __name__ == '__main__':
    test_editing_tools()
