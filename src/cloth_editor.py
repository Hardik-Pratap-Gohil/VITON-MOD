"""
Realistic cloth editing tools for VITON-HD.
Focus on colors, patterns, and textures applied to the source cloth image.
"""

import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os


class ClothColorEditor:
    """Advanced color editing for cloth images."""
    
    @staticmethod
    def adjust_hsv(cloth_tensor, cloth_mask, hue_shift=0, saturation_shift=0, brightness_shift=0):
        """
        Adjust HSV values of cloth while preserving lighting and shadows.
        
        Args:
            cloth_tensor: Cloth image tensor (B, 3, H, W) in range [-1, 1]
            cloth_mask: Cloth mask tensor (B, 1, H, W) in range [0, 1]
            hue_shift: Hue shift in degrees (-180 to 180)
            saturation_shift: Saturation shift (-100 to 100)
            brightness_shift: Brightness/Value shift (-100 to 100)
            
        Returns:
            Adjusted cloth tensor
        """
        if hue_shift == 0 and saturation_shift == 0 and brightness_shift == 0:
            return cloth_tensor
        
        device = cloth_tensor.device
        cloth_np = cloth_tensor.cpu().numpy()
        mask_np = cloth_mask.cpu().numpy()
        
        # Denormalize from [-1, 1] to [0, 255]
        cloth_np = ((cloth_np + 1) / 2.0 * 255).astype(np.uint8)
        
        batch_size = cloth_np.shape[0]
        result = np.zeros_like(cloth_np)
        
        for i in range(batch_size):
            # Convert BGR to HSV (transpose to HWC)
            img = np.transpose(cloth_np[i], (1, 2, 0))
            mask = mask_np[i, 0]
            
            # Convert to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Apply adjustments only where mask is > 0
            if hue_shift != 0:
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            
            if saturation_shift != 0:
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] + saturation_shift * 2.55, 0, 255)
            
            if brightness_shift != 0:
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] + brightness_shift * 2.55, 0, 255)
            
            # Convert back to RGB
            adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            # Apply mask: use adjusted where mask > 0, original elsewhere
            mask_3ch = np.stack([mask, mask, mask], axis=2)
            blended = (adjusted * mask_3ch + img * (1 - mask_3ch)).astype(np.uint8)
            
            result[i] = np.transpose(blended, (2, 0, 1))
        
        # Normalize back to [-1, 1]
        result = (result.astype(np.float32) / 255.0 * 2.0 - 1.0)
        return torch.from_numpy(result).to(device)
    
    @staticmethod
    def apply_color_palette(cloth_tensor, cloth_mask, palette_name='vibrant'):
        """
        Apply pre-defined color palettes.
        
        Args:
            cloth_tensor: Cloth image tensor
            cloth_mask: Cloth mask tensor
            palette_name: 'vibrant', 'pastel', 'earth', 'monochrome', 'warm', 'cool'
            
        Returns:
            Recolored cloth tensor
        """
        palettes = {
            'vibrant': {'hue': 0, 'sat': 40, 'bright': 10},
            'pastel': {'hue': 0, 'sat': -40, 'bright': 20},
            'earth': {'hue': 15, 'sat': -20, 'bright': -10},
            'monochrome': {'hue': 0, 'sat': -100, 'bright': 0},
            'warm': {'hue': 10, 'sat': 20, 'bright': 5},
            'cool': {'hue': -10, 'sat': 15, 'bright': 0},
        }
        
        params = palettes.get(palette_name, {'hue': 0, 'sat': 0, 'bright': 0})
        return ClothColorEditor.adjust_hsv(cloth_tensor, cloth_mask, 
                                          params['hue'], params['sat'], params['bright'])


class PatternGenerator:
    """Generate and apply patterns to cloth images."""
    
    @staticmethod
    def create_stripes(height, width, orientation='vertical', stripe_width=20, 
                      color1=(255, 255, 255), color2=(0, 0, 0)):
        """
        Create a striped pattern.
        
        Args:
            height, width: Pattern dimensions
            orientation: 'vertical' or 'horizontal'
            stripe_width: Width of each stripe in pixels
            color1, color2: RGB tuples for alternating colors
            
        Returns:
            Pattern image as numpy array (H, W, 3)
        """
        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        
        if orientation == 'vertical':
            for x in range(0, width, stripe_width * 2):
                pattern[:, x:x+stripe_width] = color1
                if x + stripe_width < width:
                    pattern[:, x+stripe_width:x+stripe_width*2] = color2
        else:  # horizontal
            for y in range(0, height, stripe_width * 2):
                pattern[y:y+stripe_width, :] = color1
                if y + stripe_width < height:
                    pattern[y+stripe_width:y+stripe_width*2, :] = color2
        
        return pattern
    
    @staticmethod
    def create_polkadots(height, width, dot_radius=10, spacing=40, 
                        bg_color=(255, 255, 255), dot_color=(0, 0, 0)):
        """
        Create a polkadot pattern.
        
        Args:
            height, width: Pattern dimensions
            dot_radius: Radius of each dot
            spacing: Space between dot centers
            bg_color, dot_color: RGB tuples
            
        Returns:
            Pattern image as numpy array
        """
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        for y in range(dot_radius, height, spacing):
            for x in range(dot_radius, width, spacing):
                draw.ellipse([x-dot_radius, y-dot_radius, 
                            x+dot_radius, y+dot_radius], fill=dot_color)
        
        return np.array(img)
    
    @staticmethod
    def create_checkerboard(height, width, square_size=30,
                           color1=(255, 255, 255), color2=(0, 0, 0)):
        """Create a checkerboard pattern."""
        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(0, height, square_size):
            for x in range(0, width, square_size):
                if ((y // square_size) + (x // square_size)) % 2 == 0:
                    pattern[y:y+square_size, x:x+square_size] = color1
                else:
                    pattern[y:y+square_size, x:x+square_size] = color2
        
        return pattern
    
    @staticmethod
    def apply_pattern(cloth_tensor, cloth_mask, pattern, blend_mode='multiply', opacity=0.7):
        """
        Apply a pattern to cloth while preserving lighting and texture.
        
        Args:
            cloth_tensor: Cloth image tensor (B, 3, H, W) in [-1, 1]
            cloth_mask: Cloth mask tensor (B, 1, H, W) in [0, 1]
            pattern: Pattern as numpy array (H, W, 3) in [0, 255]
            blend_mode: 'multiply', 'overlay', 'screen'
            opacity: Pattern opacity (0.0 to 1.0)
            
        Returns:
            Cloth with pattern applied
        """
        device = cloth_tensor.device
        cloth_np = cloth_tensor.cpu().numpy()
        mask_np = cloth_mask.cpu().numpy()
        
        # Denormalize cloth
        cloth_np = ((cloth_np + 1) / 2.0 * 255).astype(np.float32)
        
        batch_size, channels, height, width = cloth_np.shape
        
        # Resize pattern to match cloth
        if pattern.shape[:2] != (height, width):
            pattern = cv2.resize(pattern, (width, height), interpolation=cv2.INTER_LINEAR)
        
        pattern = pattern.astype(np.float32)
        result = np.zeros_like(cloth_np)
        
        for i in range(batch_size):
            img = np.transpose(cloth_np[i], (1, 2, 0))  # CHW -> HWC
            mask = mask_np[i, 0]
            
            # Apply blending
            if blend_mode == 'multiply':
                blended = (img * pattern) / 255.0
            elif blend_mode == 'overlay':
                # Overlay blend mode
                blended = np.where(img < 128,
                                 2 * img * pattern / 255.0,
                                 255 - 2 * (255 - img) * (255 - pattern) / 255.0)
            elif blend_mode == 'screen':
                blended = 255 - (255 - img) * (255 - pattern) / 255.0
            else:
                blended = pattern
            
            # Mix with original based on opacity
            mixed = img * (1 - opacity) + blended * opacity
            
            # Apply only to masked region
            mask_3ch = np.stack([mask, mask, mask], axis=2)
            final = mixed * mask_3ch + img * (1 - mask_3ch)
            
            result[i] = np.transpose(final, (2, 0, 1))  # HWC -> CHW
        
        # Normalize back to [-1, 1]
        result = np.clip(result, 0, 255)
        result = (result / 255.0 * 2.0 - 1.0)
        return torch.from_numpy(result.astype(np.float32)).to(device)


class LogoApplicator:
    """Apply logos and graphics to cloth."""
    
    @staticmethod
    def add_text_logo(cloth_tensor, cloth_mask, text="BRAND", position='center',
                     font_size=60, color=(255, 255, 255)):
        """
        Add text as a logo on the cloth.
        
        Args:
            cloth_tensor: Cloth image tensor (B, 3, H, W)
            cloth_mask: Cloth mask tensor (B, 1, H, W)
            text: Text to render
            position: 'center', 'top', 'bottom'
            font_size: Font size in pixels
            color: RGB tuple
            
        Returns:
            Cloth with text logo
        """
        device = cloth_tensor.device
        cloth_np = cloth_tensor.cpu().numpy()
        mask_np = cloth_mask.cpu().numpy()
        
        cloth_np = ((cloth_np + 1) / 2.0 * 255).astype(np.uint8)
        
        batch_size, channels, height, width = cloth_np.shape
        result = np.zeros_like(cloth_np)
        
        for i in range(batch_size):
            img = np.transpose(cloth_np[i], (1, 2, 0))
            mask = mask_np[i, 0]
            
            # Create PIL image
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            
            # Try to use a default font, fallback to basic
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate position
            if position == 'center':
                x = (width - text_width) // 2
                y = (height - text_height) // 2
            elif position == 'top':
                x = (width - text_width) // 2
                y = height // 4
            else:  # bottom
                x = (width - text_width) // 2
                y = 3 * height // 4
            
            # Draw text
            draw.text((x, y), text, fill=color, font=font)
            
            # Convert back
            img_with_text = np.array(pil_img)
            
            # Apply only to masked region
            mask_3ch = np.stack([mask, mask, mask], axis=2)
            final = img_with_text * mask_3ch + img * (1 - mask_3ch)
            
            result[i] = np.transpose(final, (2, 0, 1))
        
        result = (result.astype(np.float32) / 255.0 * 2.0 - 1.0)
        return torch.from_numpy(result).to(device)
    
    @staticmethod
    def add_custom_logo(cloth_tensor, cloth_mask, logo_path, position='center', 
                       scale=0.3, opacity=0.9):
        """
        Add a custom logo from an image file.
        
        Args:
            cloth_tensor: Cloth image tensor
            cloth_mask: Cloth mask tensor
            logo_path: Path to logo image (PNG with alpha channel preferred)
            position: 'center', 'top-left', 'top-right', 'bottom-left', 'bottom-right'
            scale: Logo size relative to cloth (0.0 to 1.0)
            opacity: Logo opacity (0.0 to 1.0)
            
        Returns:
            Cloth with logo
        """
        if not os.path.exists(logo_path):
            print(f"Warning: Logo file not found: {logo_path}")
            return cloth_tensor
        
        device = cloth_tensor.device
        cloth_np = cloth_tensor.cpu().numpy()
        mask_np = cloth_mask.cpu().numpy()
        
        cloth_np = ((cloth_np + 1) / 2.0 * 255).astype(np.uint8)
        
        batch_size, channels, height, width = cloth_np.shape
        
        # Load logo
        logo = Image.open(logo_path).convert('RGBA')
        logo_w, logo_h = logo.size
        
        # Resize logo
        new_w = int(width * scale)
        new_h = int(logo_h * new_w / logo_w)
        logo = logo.resize((new_w, new_h), Image.LANCZOS)
        
        result = np.zeros_like(cloth_np)
        
        for i in range(batch_size):
            img = np.transpose(cloth_np[i], (1, 2, 0))
            pil_img = Image.fromarray(img)
            
            # Calculate position
            if position == 'center':
                x, y = (width - new_w) // 2, (height - new_h) // 2
            elif position == 'top-left':
                x, y = width // 10, height // 10
            elif position == 'top-right':
                x, y = 9 * width // 10 - new_w, height // 10
            elif position == 'bottom-left':
                x, y = width // 10, 9 * height // 10 - new_h
            else:  # bottom-right
                x, y = 9 * width // 10 - new_w, 9 * height // 10 - new_h
            
            # Composite logo with opacity
            logo_copy = logo.copy()
            alpha = logo_copy.split()[3]
            alpha = Image.eval(alpha, lambda a: int(a * opacity))
            logo_copy.putalpha(alpha)
            
            pil_img.paste(logo_copy, (x, y), logo_copy)
            
            result[i] = np.transpose(np.array(pil_img), (2, 0, 1))
        
        result = (result.astype(np.float32) / 255.0 * 2.0 - 1.0)
        return torch.from_numpy(result).to(device)


class FabricSimulator:
    """Simulate different fabric textures."""
    
    @staticmethod
    def add_subtle_texture(cloth_tensor, cloth_mask, texture_type='canvas'):
        """
        Add subtle fabric texture.
        
        Args:
            cloth_tensor: Cloth image tensor
            cloth_mask: Cloth mask tensor  
            texture_type: 'canvas', 'denim', 'silk', 'linen'
            
        Returns:
            Cloth with texture
        """
        device = cloth_tensor.device
        cloth_np = cloth_tensor.cpu().numpy()
        mask_np = cloth_mask.cpu().numpy()
        
        cloth_np = ((cloth_np + 1) / 2.0 * 255).astype(np.uint8)
        
        batch_size, channels, height, width = cloth_np.shape
        result = np.zeros_like(cloth_np).astype(np.float32)
        
        for i in range(batch_size):
            img = np.transpose(cloth_np[i], (1, 2, 0)).astype(np.float32)
            mask = mask_np[i, 0]
            
            if texture_type == 'canvas':
                # Add slight grain
                noise = np.random.randint(-5, 5, (height, width, 3), dtype=np.int16)
                textured = np.clip(img + noise, 0, 255)
            
            elif texture_type == 'denim':
                # Add blue tint and grain
                noise = np.random.randint(-3, 3, (height, width, 3), dtype=np.int16)
                textured = np.clip(img + noise, 0, 255)
                # Slight blue shift
                textured[:, :, 2] = np.clip(textured[:, :, 2] * 1.05, 0, 255)
            
            elif texture_type == 'silk':
                # Smooth with slight blur
                textured = cv2.GaussianBlur(img.astype(np.uint8), (3, 3), 0).astype(np.float32)
            
            else:  # linen
                # Crosshatch pattern
                noise = np.random.randint(-4, 4, (height, width, 3), dtype=np.int16)
                textured = np.clip(img + noise, 0, 255)
            
            # Apply only to masked region
            mask_3ch = np.stack([mask, mask, mask], axis=2)
            final = textured * mask_3ch + img * (1 - mask_3ch)
            
            result[i] = np.transpose(final, (2, 0, 1))
        
        result = np.clip(result, 0, 255)
        result = (result / 255.0 * 2.0 - 1.0)
        return torch.from_numpy(result.astype(np.float32)).to(device)
