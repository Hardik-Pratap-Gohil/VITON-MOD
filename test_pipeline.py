"""
Quick test script for VITONInference class
"""

import torch
from inference_pipeline import VITONInference
from datasets import VITONDataset, VITONDataLoader
import argparse

# Create dummy options for dataset
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

def test_inference():
    print("Testing VITONInference class...")
    
    # Initialize pipeline
    pipeline = VITONInference(checkpoint_dir='./checkpoints/')
    
    # Load models
    pipeline.load_models()
    
    # Load test data
    opt = TestOpt()
    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)
    
    # Get first batch
    inputs = next(iter(test_loader.data_loader))
    
    print(f"\nProcessing: {inputs['img_name'][0]} + {inputs['c_name']['unpaired'][0]}")
    
    # Run pipeline with intermediates
    output, intermediates = pipeline.run_full_pipeline(
        inputs['img_agnostic'],
        inputs['parse_agnostic'],
        inputs['pose'],
        inputs['cloth']['unpaired'],
        inputs['cloth_mask']['unpaired'],
        return_intermediates=True
    )
    
    print("\n✓ Pipeline executed successfully!")
    print(f"  Output shape: {output.shape}")
    print(f"  Warped cloth shape: {intermediates['warped_c'].shape}")
    print(f"  Parse shape: {intermediates['parse'].shape}")
    
    # Test editing workflow
    print("\nTesting edit workflow...")
    edited_warped_c = intermediates['warped_c'] * 0.8  # Simulate darkening
    
    output_edited = pipeline.run_with_edited_cloth(intermediates, edited_warped_c)
    print(f"✓ Edited output shape: {output_edited.shape}")
    
    print("\n✅ All tests passed!")

if __name__ == '__main__':
    test_inference()
