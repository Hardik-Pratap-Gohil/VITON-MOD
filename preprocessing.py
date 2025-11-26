"""
Preprocessing utilities for loading and preparing images for VITON-HD pipeline.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


class VITONPreprocessor:
    """
    Handles loading and preprocessing of images for the VITON-HD pipeline.
    """
    
    def __init__(self, dataset_dir='./datasets/', load_height=1024, load_width=768):
        """
        Initialize preprocessor.
        
        Args:
            dataset_dir: Path to dataset directory
            load_height: Target image height
            load_width: Target image width
        """
        self.dataset_dir = dataset_dir
        self.load_height = load_height
        self.load_width = load_width
        
        # Standard VITON-HD transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def load_person_data(self, img_name, mode='test'):
        """
        Load all data for a person image.
        
        Args:
            img_name: Person image filename (e.g., '08909_00.jpg')
            mode: 'test' or 'train'
            
        Returns:
            dict with img, img_agnostic, parse_agnostic, pose tensors
        """
        data_path = os.path.join(self.dataset_dir, mode)
        
        # Load person image
        img = Image.open(os.path.join(data_path, 'image', img_name))
        img = transforms.Resize(self.load_width, interpolation=2)(img)
        
        # Load pose
        pose_name = img_name.replace('.jpg', '_rendered.png')
        pose_rgb = Image.open(os.path.join(data_path, 'openpose-img', pose_name))
        pose_rgb = transforms.Resize(self.load_width, interpolation=2)(pose_rgb)
        
        # Load pose keypoints
        pose_json_name = img_name.replace('.jpg', '_keypoints.json')
        with open(os.path.join(data_path, 'openpose-json', pose_json_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data).reshape((-1, 3))[:, :2]
        
        # Load parse
        parse_name = img_name.replace('.jpg', '.png')
        parse = Image.open(os.path.join(data_path, 'image-parse', parse_name))
        parse = transforms.Resize(self.load_width, interpolation=0)(parse)
        
        # Generate agnostic representations
        img_agnostic = self._get_img_agnostic(img, parse, pose_data)
        parse_agnostic = self._get_parse_agnostic(parse, pose_data)
        
        # Convert to tensors
        img_tensor = self.transform(img).unsqueeze(0)
        img_agnostic_tensor = self.transform(img_agnostic).unsqueeze(0)
        pose_tensor = self.transform(pose_rgb).unsqueeze(0)
        
        parse_agnostic_tensor = torch.from_numpy(np.array(parse_agnostic)[None]).long()
        parse_agnostic_map = self._convert_parse_to_map(parse_agnostic_tensor)
        
        return {
            'img': img_tensor,
            'img_agnostic': img_agnostic_tensor,
            'parse_agnostic': parse_agnostic_map,
            'pose': pose_tensor,
            'img_name': img_name
        }
    
    def load_cloth_data(self, cloth_name, mode='test'):
        """
        Load cloth image and mask.
        
        Args:
            cloth_name: Cloth image filename (e.g., '02783_00.jpg')
            mode: 'test' or 'train'
            
        Returns:
            dict with cloth tensor and cloth_mask tensor
        """
        data_path = os.path.join(self.dataset_dir, mode)
        
        # Load cloth
        c = Image.open(os.path.join(data_path, 'cloth', cloth_name)).convert('RGB')
        c = transforms.Resize(self.load_width, interpolation=2)(c)
        c_tensor = self.transform(c).unsqueeze(0)
        
        # Load cloth mask
        cm = Image.open(os.path.join(data_path, 'cloth-mask', cloth_name))
        cm = transforms.Resize(self.load_width, interpolation=0)(cm)
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm_tensor = torch.from_numpy(cm_array).unsqueeze(0).unsqueeze(0)
        
        return {
            'cloth': c_tensor,
            'cloth_mask': cm_tensor,
            'cloth_name': cloth_name
        }
    
    def _get_parse_agnostic(self, parse, pose_data):
        """Generate agnostic parse map (removes upper body clothing)."""
        from PIL import ImageDraw
        
        parse_array = np.array(parse)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)
        
        r = 10
        agnostic = parse.copy()
        
        # Mask arms
        for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
            mask_arm = Image.new('L', (self.load_width, self.load_height), 'black')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            i_prev = pose_ids[0]
            for i in pose_ids[1:]:
                if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or \
                   (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
                pointx, pointy = pose_data[i]
                radius = r*4 if i == pose_ids[-1] else r*15
                mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 
                                     'white', 'white')
                i_prev = i
            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))
        
        # Mask torso & neck
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))
        
        return agnostic
    
    def _get_img_agnostic(self, img, parse, pose_data):
        """Generate agnostic person image (removes upper body appearance)."""
        from PIL import ImageDraw
        
        parse_array = np.array(parse)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 16).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))
        
        r = 20
        agnostic = img.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)
        
        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
        
        # Mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or \
               (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        
        # Mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')
        
        # Mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
        
        return agnostic
    
    def _convert_parse_to_map(self, parse_agnostic_tensor):
        """Convert parse labels to multi-channel map."""
        labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }
        
        parse_agnostic_map = torch.zeros(20, self.load_height, self.load_width, dtype=torch.float)
        parse_agnostic_map.scatter_(0, parse_agnostic_tensor[0].unsqueeze(0), 1.0)
        
        new_parse_agnostic_map = torch.zeros(13, self.load_height, self.load_width, dtype=torch.float)
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]
        
        return new_parse_agnostic_map.unsqueeze(0)
    
    def get_available_images(self, mode='test'):
        """
        Get list of available person and cloth images.
        
        Args:
            mode: 'test' or 'train'
            
        Returns:
            tuple of (person_images, cloth_images)
        """
        data_path = os.path.join(self.dataset_dir, mode)
        
        person_images = sorted(os.listdir(os.path.join(data_path, 'image')))
        cloth_images = sorted(os.listdir(os.path.join(data_path, 'cloth')))
        
        return person_images, cloth_images
