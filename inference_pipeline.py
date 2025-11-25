"""
VITON-HD Inference Pipeline
Modular class for running virtual try-on inference with intermediate output access.
"""

import os
import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm
from PIL import Image
import numpy as np

from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint


class VITONInference:
    """
    Modular inference pipeline for VITON-HD.
    Exposes intermediate outputs for editing and manipulation.
    """
    
    def __init__(self, checkpoint_dir='./checkpoints/', device=None):
        """
        Initialize the VITON-HD inference pipeline.
        
        Args:
            checkpoint_dir: Path to checkpoint directory
            device: torch device (auto-detects if None)
        """
        self.checkpoint_dir = checkpoint_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model configurations
        self.load_height = 1024
        self.load_width = 768
        self.semantic_nc = 13
        
        # Initialize models (lazy loading)
        self.seg = None
        self.gmm = None
        self.alias = None
        
        # Cache for intermediate outputs
        self.cache = {}
        
        print(f"VITONInference initialized on device: {self.device}")
    
    def load_models(self, seg_checkpoint='seg_final.pth', 
                    gmm_checkpoint='gmm_final.pth',
                    alias_checkpoint='alias_final.pth'):
        """
        Load pretrained models from checkpoints.
        
        Args:
            seg_checkpoint: Segmentation generator checkpoint
            gmm_checkpoint: GMM checkpoint
            alias_checkpoint: ALIAS generator checkpoint
        """
        print("Loading models...")
        
        # Create models
        class DummyOpt:
            def __init__(self):
                self.semantic_nc = 13
                self.init_type = 'xavier'
                self.init_variance = 0.02
                self.grid_size = 5
                self.norm_G = 'spectralaliasinstance'
                self.ngf = 64
                self.num_upsampling_layers = 'most'
                self.load_height = 1024
                self.load_width = 768
        
        opt = DummyOpt()
        
        # Initialize models
        self.seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
        self.gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
        opt.semantic_nc = 7
        self.alias = ALIASGenerator(opt, input_nc=9)
        opt.semantic_nc = 13
        
        # Load checkpoints
        load_checkpoint(self.seg, os.path.join(self.checkpoint_dir, seg_checkpoint))
        load_checkpoint(self.gmm, os.path.join(self.checkpoint_dir, gmm_checkpoint))
        load_checkpoint(self.alias, os.path.join(self.checkpoint_dir, alias_checkpoint))
        
        # Move to device and set to eval mode
        self.seg.to(self.device).eval()
        self.gmm.to(self.device).eval()
        self.alias.to(self.device).eval()
        
        print("Models loaded successfully!")
    
    def set_resolution(self, height, width):
        """
        Change the working resolution (for preview vs HD rendering).
        
        Args:
            height: Target height
            width: Target width
        """
        self.load_height = height
        self.load_width = width
        print(f"Resolution set to {width}x{height}")
    
    def run_segmentation(self, parse_agnostic, pose, c, cm):
        """
        Run segmentation generation (Part 1 of pipeline).
        
        Args:
            parse_agnostic: Agnostic parse map
            pose: Pose map
            c: Cloth image
            cm: Cloth mask
            
        Returns:
            parse: Generated segmentation map
        """
        up = nn.Upsample(size=(self.load_height, self.load_width), mode='bilinear')
        gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).to(self.device)
        
        # Downsample inputs
        parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
        pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
        c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
        cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
        
        # Create segmentation input
        seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, 
                              pose_down, gen_noise(cm_down.size()).to(self.device)), dim=1)
        
        # Generate segmentation
        with torch.no_grad():
            parse_pred_down = self.seg(seg_input)
            parse_pred = gauss(up(parse_pred_down))
            parse_pred = parse_pred.argmax(dim=1)[:, None]
        
        # Convert to one-hot encoding
        parse_old = torch.zeros(parse_pred.size(0), 13, self.load_height, 
                               self.load_width, dtype=torch.float).to(self.device)
        parse_old.scatter_(1, parse_pred, 1.0)
        
        # Merge labels
        labels = {
            0: ['background', [0]],
            1: ['paste', [2, 4, 7, 8, 9, 10, 11]],
            2: ['upper', [3]],
            3: ['hair', [1]],
            4: ['left_arm', [5]],
            5: ['right_arm', [6]],
            6: ['noise', [12]]
        }
        
        parse = torch.zeros(parse_pred.size(0), 7, self.load_height, 
                           self.load_width, dtype=torch.float).to(self.device)
        for j in range(len(labels)):
            for label in labels[j][1]:
                parse[:, j] += parse_old[:, label]
        
        return parse
    
    def run_gmm(self, img_agnostic, parse, pose, c, cm):
        """
        Run Geometric Matching Module (Part 2 of pipeline).
        
        Args:
            img_agnostic: Agnostic person image
            parse: Segmentation map
            pose: Pose map
            c: Cloth image
            cm: Cloth mask
            
        Returns:
            warped_c: Warped cloth image
            warped_cm: Warped cloth mask
            warped_grid: Warping grid (for debugging)
        """
        # Prepare GMM inputs
        agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
        parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
        pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
        c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
        
        gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)
        
        # Run GMM
        with torch.no_grad():
            _, warped_grid = self.gmm(gmm_input, c_gmm)
            warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')
        
        return warped_c, warped_cm, warped_grid
    
    def run_alias(self, img_agnostic, pose, warped_c, parse, warped_cm):
        """
        Run ALIAS Generator (Part 3 of pipeline).
        
        Args:
            img_agnostic: Agnostic person image
            pose: Pose map
            warped_c: Warped cloth image
            parse: Segmentation map
            warped_cm: Warped cloth mask
            
        Returns:
            output: Final try-on image
        """
        # Compute misalignment mask
        misalign_mask = parse[:, 2:3] - warped_cm
        misalign_mask[misalign_mask < 0.0] = 0.0
        
        # Create parse_div
        parse_div = torch.cat((parse, misalign_mask), dim=1)
        parse_div[:, 2:3] -= misalign_mask
        
        # Run ALIAS generator
        with torch.no_grad():
            output = self.alias(torch.cat((img_agnostic, pose, warped_c), dim=1), 
                              parse, parse_div, misalign_mask)
        
        return output
    
    def run_full_pipeline(self, img_agnostic, parse_agnostic, pose, c, cm, 
                         return_intermediates=False):
        """
        Run the complete VITON-HD pipeline.
        
        Args:
            img_agnostic: Agnostic person image (tensor)
            parse_agnostic: Agnostic parse map (tensor)
            pose: Pose map (tensor)
            c: Cloth image (tensor)
            cm: Cloth mask (tensor)
            return_intermediates: If True, return intermediate outputs
            
        Returns:
            output: Final try-on image
            intermediates: Dict of intermediate outputs (if return_intermediates=True)
        """
        if self.seg is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Move inputs to device
        img_agnostic = img_agnostic.to(self.device)
        parse_agnostic = parse_agnostic.to(self.device)
        pose = pose.to(self.device)
        c = c.to(self.device)
        cm = cm.to(self.device)
        
        # Part 1: Segmentation
        parse = self.run_segmentation(parse_agnostic, pose, c, cm)
        
        # Part 2: GMM
        warped_c, warped_cm, warped_grid = self.run_gmm(img_agnostic, parse, pose, c, cm)
        
        # Part 3: ALIAS
        output = self.run_alias(img_agnostic, pose, warped_c, parse, warped_cm)
        
        if return_intermediates:
            intermediates = {
                'parse': parse,
                'warped_c': warped_c,
                'warped_cm': warped_cm,
                'warped_grid': warped_grid,
                'img_agnostic': img_agnostic,
                'pose': pose
            }
            return output, intermediates
        
        return output
    
    def run_with_edited_cloth(self, intermediates, edited_warped_c, edited_warped_cm=None):
        """
        Run ALIAS generator with edited warped cloth.
        Useful for applying color/texture changes without re-running GMM.
        
        Args:
            intermediates: Dict from run_full_pipeline(return_intermediates=True)
            edited_warped_c: Modified warped cloth
            edited_warped_cm: Modified warped cloth mask (optional)
            
        Returns:
            output: Final try-on image with edited cloth
        """
        warped_cm = edited_warped_cm if edited_warped_cm is not None else intermediates['warped_cm']
        
        output = self.run_alias(
            intermediates['img_agnostic'],
            intermediates['pose'],
            edited_warped_c,
            intermediates['parse'],
            warped_cm
        )
        
        return output
