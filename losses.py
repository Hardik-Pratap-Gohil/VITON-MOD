"""
Loss functions for Frequency-Enhanced VITON-HD
Includes novel frequency-domain losses for detail preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FrequencyDomainLoss(nn.Module):
    """
    NOVEL: Frequency-domain loss for preserving cloth details.
    Computes loss in DCT domain with higher weight on high frequencies.
    """
    def __init__(self, high_freq_weight=2.0):
        super(FrequencyDomainLoss, self).__init__()
        self.high_freq_weight = high_freq_weight

    def dct_2d(self, x):
        """2D DCT using FFT"""
        X = torch.fft.rfft2(x, norm='ortho')
        x_dct = torch.fft.irfft2(X, s=x.shape[-2:], norm='ortho')
        return x_dct

    def create_freq_weight_mask(self, H, W, device):
        """
        Create weighting mask that emphasizes high frequencies.
        High frequencies (details) get higher weight in loss.
        """
        mask = torch.ones(1, 1, H, W, device=device)
        center_h, center_w = H // 2, W // 2

        # Gradually increase weight away from center (DC component)
        for i in range(H):
            for j in range(W):
                dist = ((i - center_h)**2 + (j - center_w)**2) ** 0.5
                max_dist = ((H/2)**2 + (W/2)**2) ** 0.5
                # Linear weight from 1.0 (center) to high_freq_weight (edges)
                mask[0, 0, i, j] = 1.0 + (self.high_freq_weight - 1.0) * (dist / max_dist)

        return mask

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
        Returns:
            Frequency-domain weighted loss
        """
        B, C, H, W = pred.shape
        device = pred.device

        # Create frequency weight mask
        weight_mask = self.create_freq_weight_mask(H, W, device)

        total_loss = 0.0
        for c in range(C):
            # Transform to frequency domain
            pred_freq = self.dct_2d(pred[:, c:c+1])
            target_freq = self.dct_2d(target[:, c:c+1])

            # Weighted L1 loss in frequency domain
            freq_diff = torch.abs(pred_freq - target_freq)
            weighted_diff = freq_diff * weight_mask
            total_loss += weighted_diff.mean()

        return total_loss / C


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    Standard in image synthesis but important for cloth detail quality.
    """
    def __init__(self, use_gpu=True):
        super(PerceptualLoss, self).__init__()

        # Load pre-trained VGG19
        vgg = torchvision.models.vgg19(pretrained=True).features

        # Extract feature blocks
        self.block1 = vgg[:4].eval()    # relu1_2
        self.block2 = vgg[4:9].eval()   # relu2_2
        self.block3 = vgg[9:18].eval()  # relu3_4
        self.block4 = vgg[18:27].eval() # relu4_4

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image [B, 3, H, W] in range [-1, 1]
            target: Target image [B, 3, H, W] in range [-1, 1]
        Returns:
            Multi-scale perceptual loss
        """
        # Normalize from [-1, 1] to [0, 1]
        pred = (pred + 1) / 2
        target = (target + 1) / 2

        # VGG normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        pred = (pred - mean) / std
        target = (target - mean) / std

        # Compute features and loss at multiple scales
        loss = 0.0

        pred_feat = self.block1(pred)
        target_feat = self.block1(target)
        loss += F.l1_loss(pred_feat, target_feat)

        pred_feat = self.block2(pred_feat)
        target_feat = self.block2(target_feat)
        loss += F.l1_loss(pred_feat, target_feat)

        pred_feat = self.block3(pred_feat)
        target_feat = self.block3(target_feat)
        loss += F.l1_loss(pred_feat, target_feat)

        pred_feat = self.block4(pred_feat)
        target_feat = self.block4(target_feat)
        loss += F.l1_loss(pred_feat, target_feat)

        return loss


class StyleLoss(nn.Module):
    """
    Style loss using Gram matrices.
    Helps preserve cloth texture patterns.
    """
    def __init__(self):
        super(StyleLoss, self).__init__()

        # Use VGG19 for style features
        vgg = torchvision.models.vgg19(pretrained=True).features
        self.block1 = vgg[:4].eval()
        self.block2 = vgg[4:9].eval()
        self.block3 = vgg[9:18].eval()

        for param in self.parameters():
            param.requires_grad = False

    def gram_matrix(self, x):
        """Compute Gram matrix for style representation"""
        B, C, H, W = x.size()
        features = x.view(B, C, H * W)
        G = torch.bmm(features, features.transpose(1, 2))
        return G / (C * H * W)

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        Returns:
            Style loss
        """
        # Normalize
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        pred = (pred - mean) / std
        target = (target - mean) / std

        loss = 0.0

        # Compute style loss at multiple scales
        pred_feat = self.block1(pred)
        target_feat = self.block1(target)
        loss += F.l1_loss(self.gram_matrix(pred_feat), self.gram_matrix(target_feat))

        pred_feat = self.block2(pred_feat)
        target_feat = self.block2(target_feat)
        loss += F.l1_loss(self.gram_matrix(pred_feat), self.gram_matrix(target_feat))

        pred_feat = self.block3(pred_feat)
        target_feat = self.block3(target_feat)
        loss += F.l1_loss(self.gram_matrix(pred_feat), self.gram_matrix(target_feat))

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for training Frequency-Enhanced VITON.

    Total Loss = L1 + λ_freq * FreqLoss + λ_perc * PercLoss + λ_style * StyleLoss
    """
    def __init__(self,
                 lambda_l1=1.0,
                 lambda_freq=0.5,
                 lambda_perceptual=0.1,
                 lambda_style=0.05,
                 use_gpu=True):
        super(CombinedLoss, self).__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_freq = lambda_freq
        self.lambda_perceptual = lambda_perceptual
        self.lambda_style = lambda_style

        # Initialize loss modules
        self.freq_loss = FrequencyDomainLoss(high_freq_weight=2.0)
        self.perceptual_loss = PerceptualLoss(use_gpu=use_gpu)
        self.style_loss = StyleLoss()

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        Returns:
            Dictionary with total loss and individual loss components
        """
        # L1 loss (spatial domain)
        l1_loss = F.l1_loss(pred, target)

        # Frequency-domain loss (NOVEL)
        freq_loss = self.freq_loss(pred, target)

        # Perceptual loss
        perc_loss = self.perceptual_loss(pred, target)

        # Style loss
        style_loss = self.style_loss(pred, target)

        # Total loss
        total_loss = (self.lambda_l1 * l1_loss +
                     self.lambda_freq * freq_loss +
                     self.lambda_perceptual * perc_loss +
                     self.lambda_style * style_loss)

        # Return breakdown for logging
        return {
            'total': total_loss,
            'l1': l1_loss,
            'frequency': freq_loss,
            'perceptual': perc_loss,
            'style': style_loss
        }


class SegmentationLoss(nn.Module):
    """Cross-entropy loss for segmentation generator"""
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted segmentation [B, num_classes, H, W]
            target: Target segmentation [B, H, W] (class indices)
        Returns:
            Cross-entropy loss
        """
        return self.criterion(pred, target)
