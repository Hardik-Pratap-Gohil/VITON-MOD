"""
Editing Tools for VITON-HD
Provides color/texture manipulation and mask morphology operations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class ColorEditor:
    """
    Color and texture editing tools for warped clothing.
    All operations preserve tensor format for pipeline compatibility.
    """
    
    @staticmethod
    def adjust_hsv(cloth_tensor, hue_shift=0, saturation_shift=0, brightness_shift=0):
        """
        Adjust HSV values of cloth tensor.
        
        Args:
            cloth_tensor: Cloth image tensor (B, 3, H, W) in range [-1, 1]
            hue_shift: Hue shift in degrees (-180 to 180)
            saturation_shift: Saturation shift in percentage (-100 to 100)
            brightness_shift: Brightness shift in percentage (-50 to 50)
            
        Returns:
            Adjusted cloth tensor in range [-1, 1]
        """
        if hue_shift == 0 and saturation_shift == 0 and brightness_shift == 0:
            return cloth_tensor
        
        # Convert from [-1, 1] to [0, 1]
        cloth = (cloth_tensor + 1) / 2.0
        
        # Move to CPU and convert to numpy for HSV operations
        device = cloth.device
        cloth_np = cloth.cpu().numpy()
        
        # Process each image in batch
        batch_size = cloth_np.shape[0]
        result = np.zeros_like(cloth_np)
        
        for i in range(batch_size):
            # Convert from (C, H, W) to (H, W, C)
            img = np.transpose(cloth_np[i], (1, 2, 0))
            img = (img * 255).astype(np.uint8)
            
            # Convert to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Adjust Hue (0-180 in OpenCV)
            if hue_shift != 0:
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift / 2) % 180
            
            # Adjust Saturation (0-255)
            if saturation_shift != 0:
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + saturation_shift / 100.0), 0, 255)
            
            # Adjust Brightness/Value (0-255)
            if brightness_shift != 0:
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + brightness_shift / 100.0), 0, 255)
            
            # Convert back to RGB
            hsv = hsv.astype(np.uint8)
            img_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # Convert back to (C, H, W) and normalize to [0, 1]
            img_adjusted = img_adjusted.astype(np.float32) / 255.0
            result[i] = np.transpose(img_adjusted, (2, 0, 1))
        
        # Convert back to tensor and scale to [-1, 1]
        result_tensor = torch.from_numpy(result).to(device)
        result_tensor = result_tensor * 2.0 - 1.0
        
        return result_tensor
    
    @staticmethod
    def apply_pattern(cloth_tensor, pattern_image, opacity=0.5, blend_mode='multiply'):
        """
        Apply a pattern overlay to the cloth.
        
        Args:
            cloth_tensor: Cloth image tensor (B, 3, H, W) in range [-1, 1]
            pattern_image: Pattern as PIL Image or numpy array
            opacity: Pattern opacity (0 to 1)
            blend_mode: 'multiply', 'overlay', or 'screen'
            
        Returns:
            Cloth with pattern applied
        """
        if opacity == 0:
            return cloth_tensor
        
        # Convert pattern to tensor
        if isinstance(pattern_image, Image.Image):
            pattern_np = np.array(pattern_image).astype(np.float32) / 255.0
        else:
            pattern_np = pattern_image.astype(np.float32) / 255.0
        
        # Ensure RGB
        if len(pattern_np.shape) == 2:
            pattern_np = np.stack([pattern_np] * 3, axis=-1)
        elif pattern_np.shape[-1] == 4:
            pattern_np = pattern_np[:, :, :3]
        
        # Resize pattern to match cloth
        device = cloth_tensor.device
        _, _, h, w = cloth_tensor.shape
        pattern_pil = Image.fromarray((pattern_np * 255).astype(np.uint8))
        pattern_pil = pattern_pil.resize((w, h), Image.BILINEAR)
        pattern_np = np.array(pattern_pil).astype(np.float32) / 255.0
        
        # Convert to tensor
        pattern_tensor = torch.from_numpy(pattern_np).permute(2, 0, 1).unsqueeze(0).to(device)
        pattern_tensor = pattern_tensor * 2.0 - 1.0  # Scale to [-1, 1]
        
        # Broadcast to batch size
        batch_size = cloth_tensor.shape[0]
        pattern_tensor = pattern_tensor.repeat(batch_size, 1, 1, 1)
        
        # Apply blend mode
        if blend_mode == 'multiply':
            # Convert to [0, 1] for multiply
            cloth_01 = (cloth_tensor + 1) / 2.0
            pattern_01 = (pattern_tensor + 1) / 2.0
            blended = cloth_01 * pattern_01
            blended = blended * 2.0 - 1.0  # Back to [-1, 1]
        elif blend_mode == 'overlay':
            cloth_01 = (cloth_tensor + 1) / 2.0
            pattern_01 = (pattern_tensor + 1) / 2.0
            blended = torch.where(cloth_01 < 0.5,
                                 2 * cloth_01 * pattern_01,
                                 1 - 2 * (1 - cloth_01) * (1 - pattern_01))
            blended = blended * 2.0 - 1.0
        elif blend_mode == 'screen':
            cloth_01 = (cloth_tensor + 1) / 2.0
            pattern_01 = (pattern_tensor + 1) / 2.0
            blended = 1 - (1 - cloth_01) * (1 - pattern_01)
            blended = blended * 2.0 - 1.0
        else:
            blended = pattern_tensor
        
        # Mix with original based on opacity
        result = cloth_tensor * (1 - opacity) + blended * opacity
        
        return result


class MaskMorpher:
    """
    Mask morphology operations for fit and style adjustments.
    """
    
    @staticmethod
    def adjust_fit(mask_tensor, fit_percentage=0):
        """
        Adjust garment fit by dilating (looser) or eroding (tighter) the mask.
        
        Args:
            mask_tensor: Mask tensor (B, 1, H, W) in range [0, 1]
            fit_percentage: Fit adjustment (-15 to 15)
                          Negative = tighter, Positive = looser
            
        Returns:
            Adjusted mask tensor
        """
        if fit_percentage == 0:
            return mask_tensor
        
        device = mask_tensor.device
        batch_size, channels, height, width = mask_tensor.shape
        
        # Convert to numpy for morphological operations
        mask_np = mask_tensor.cpu().numpy()
        result = np.zeros_like(mask_np)
        
        # Calculate kernel size based on percentage
        kernel_size = max(1, int(abs(fit_percentage) / 5 * 3))  # 1-9 pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        for i in range(batch_size):
            for c in range(channels):
                mask_img = (mask_np[i, c] * 255).astype(np.uint8)
                
                if fit_percentage > 0:
                    # Dilate for looser fit
                    iterations = max(1, int(fit_percentage / 5))
                    adjusted = cv2.dilate(mask_img, kernel, iterations=iterations)
                else:
                    # Erode for tighter fit
                    iterations = max(1, int(abs(fit_percentage) / 5))
                    adjusted = cv2.erode(mask_img, kernel, iterations=iterations)
                
                result[i, c] = adjusted.astype(np.float32) / 255.0
        
        result_tensor = torch.from_numpy(result).to(device)
        return result_tensor
    
    @staticmethod
    def adjust_sleeve_length(parse_tensor, warped_cm, shift_pixels=0):
        """
        Adjust sleeve length by shifting the sleeve region vertically.
        
        Args:
            parse_tensor: Segmentation tensor (B, 7, H, W)
            warped_cm: Warped cloth mask (B, 1, H, W)
            shift_pixels: Pixels to shift (-20 to 20)
                         Negative = shorter, Positive = longer
            
        Returns:
            Adjusted parse tensor, adjusted cloth mask
        """
        if shift_pixels == 0:
            return parse_tensor, warped_cm
        
        device = parse_tensor.device
        
        # Clone tensors to avoid in-place modification
        adjusted_parse = parse_tensor.clone()
        adjusted_cm = warped_cm.clone()
        
        # Shift left arm (channel 4) and right arm (channel 5)
        if shift_pixels != 0:
            # Create shifted versions
            left_arm = adjusted_parse[:, 4:5, :, :]
            right_arm = adjusted_parse[:, 5:6, :, :]
            
            # Shift vertically (positive = down/longer, negative = up/shorter)
            if shift_pixels > 0:
                # Extend sleeves downward
                left_arm_shifted = F.pad(left_arm[:, :, :-shift_pixels, :], 
                                        (0, 0, shift_pixels, 0))
                right_arm_shifted = F.pad(right_arm[:, :, :-shift_pixels, :], 
                                         (0, 0, shift_pixels, 0))
            else:
                # Shorten sleeves (shift up)
                shift_abs = abs(shift_pixels)
                left_arm_shifted = F.pad(left_arm[:, :, shift_abs:, :], 
                                        (0, 0, 0, shift_abs))
                right_arm_shifted = F.pad(right_arm[:, :, shift_abs:, :], 
                                         (0, 0, 0, shift_abs))
            
            adjusted_parse[:, 4:5, :, :] = left_arm_shifted
            adjusted_parse[:, 5:6, :, :] = right_arm_shifted
        
        return adjusted_parse, adjusted_cm
    
    @staticmethod
    def apply_fit_preset(mask_tensor, preset_name='Original'):
        """
        Apply predefined fit presets.
        
        Args:
            mask_tensor: Mask tensor (B, 1, H, W)
            preset_name: 'Original', 'Slim', 'Fitted', 'Relaxed', 'Oversized'
            
        Returns:
            Adjusted mask tensor
        """
        preset_map = {
            'Original': 0,
            'Slim (-10%)': -10,
            'Fitted (-5%)': -5,
            'Relaxed (+5%)': 5,
            'Oversized (+10%)': 10
        }
        
        fit_percentage = preset_map.get(preset_name, 0)
        return MaskMorpher.adjust_fit(mask_tensor, fit_percentage)
