"""
Quick demo to test the Frequency-Aware modules
This script verifies that the new modules work correctly
"""

import torch
from networks import FrequencyAwareClothEncoder, FrequencyEnhancedGMM
from losses import FrequencyDomainLoss, CombinedLoss

print("="*60)
print("Testing Frequency-Domain Modules")
print("="*60)

# Test 1: FrequencyAwareClothEncoder
print("\n[1/3] Testing FrequencyAwareClothEncoder...")
encoder = FrequencyAwareClothEncoder(input_nc=3, output_nc=64)
test_cloth = torch.randn(2, 3, 256, 192)  # Batch of 2, RGB, 256x192
output = encoder(test_cloth)
print(f"✓ Input shape: {test_cloth.shape}")
print(f"✓ Output shape: {output.shape}")
print(f"✓ Expected: torch.Size([2, 64, 256, 192])")
assert output.shape == torch.Size([2, 64, 256, 192]), "Shape mismatch!"
print("✓ FrequencyAwareClothEncoder works!")

# Test 2: FrequencyDomainLoss
print("\n[2/3] Testing FrequencyDomainLoss...")
freq_loss = FrequencyDomainLoss(high_freq_weight=2.0)
pred = torch.randn(2, 3, 256, 192)
target = torch.randn(2, 3, 256, 192)
loss = freq_loss(pred, target)
print(f"✓ Loss value: {loss.item():.4f}")
print(f"✓ Loss shape: {loss.shape}")
assert loss.numel() == 1, "Loss should be scalar!"
print("✓ FrequencyDomainLoss works!")

# Test 3: CombinedLoss
print("\n[3/3] Testing CombinedLoss...")
combined_loss = CombinedLoss(
    lambda_l1=1.0,
    lambda_freq=0.5,
    lambda_perceptual=0.1,
    lambda_style=0.05,
    use_gpu=False
)
pred = torch.randn(2, 3, 256, 192) * 2 - 1  # Range [-1, 1]
target = torch.randn(2, 3, 256, 192) * 2 - 1
loss_dict = combined_loss(pred, target)
print(f"✓ Total loss: {loss_dict['total'].item():.4f}")
print(f"✓ L1 loss: {loss_dict['l1'].item():.4f}")
print(f"✓ Frequency loss: {loss_dict['frequency'].item():.4f}")
print(f"✓ Perceptual loss: {loss_dict['perceptual'].item():.4f}")
print(f"✓ Style loss: {loss_dict['style'].item():.4f}")
print("✓ CombinedLoss works!")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\nYour frequency-domain modules are working correctly!")
print("You can now proceed to training with:")
print("  python train.py --name my_experiment --train_mode gmm")
print("="*60)
