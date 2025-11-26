"""
VITON-MOD: Interactive Cloth Editor for VITON-HD
Core source modules for cloth editing and virtual try-on.
"""

__version__ = "1.0.0"

from .cloth_editor import ClothColorEditor, PatternGenerator, LogoApplicator, FabricSimulator
from .inference_pipeline import VITONInference
from .preprocessing import VITONPreprocessor

__all__ = [
    'ClothColorEditor',
    'PatternGenerator', 
    'LogoApplicator',
    'FabricSimulator',
    'VITONInference',
    'VITONPreprocessor'
]
