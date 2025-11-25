"""
VITON-HD Interactive Editor - Streamlit Web Application
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import time

from inference_pipeline import VITONInference
from editing_tools import ColorEditor, MaskMorpher
from preprocessing import VITONPreprocessor
from utils import save_images


# Page configuration
st.set_page_config(
    page_title="VITON-HD Interactive Editor",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load and cache the VITON pipeline."""
    pipeline = VITONInference(checkpoint_dir='./checkpoints/')
    pipeline.load_models()
    return pipeline


@st.cache_resource
def load_preprocessor():
    """Load and cache the preprocessor."""
    return VITONPreprocessor(dataset_dir='./datasets/')


def tensor_to_pil(tensor):
    """Convert tensor to PIL Image."""
    # Convert from [-1, 1] to [0, 255]
    array = ((tensor.clone() + 1) * 0.5 * 255).cpu().clamp(0, 255)
    array = array.squeeze(0).permute(1, 2, 0).numpy().astype('uint8')
    return Image.fromarray(array)


def pil_to_bytes(pil_img):
    """Convert PIL Image to bytes for download."""
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    return buf.getvalue()


def initialize_session_state():
    """Initialize session state variables."""
    if 'person_image' not in st.session_state:
        st.session_state.person_image = None
    if 'cloth_image' not in st.session_state:
        st.session_state.cloth_image = None
    if 'preview_output' not in st.session_state:
        st.session_state.preview_output = None
    if 'hd_output' not in st.session_state:
        st.session_state.hd_output = None
    if 'intermediates' not in st.session_state:
        st.session_state.intermediates = None
    if 'person_data' not in st.session_state:
        st.session_state.person_data = None
    if 'cloth_data' not in st.session_state:
        st.session_state.cloth_data = None


def reset_all_params():
    """Reset all editing parameters to default."""
    st.session_state.hue_shift = 0
    st.session_state.saturation_shift = 0
    st.session_state.brightness_shift = 0
    st.session_state.fit_preset = 'Original'
    st.session_state.sleeve_shift = 0
    st.session_state.preview_output = None
    st.session_state.hd_output = None


def reset_color_params():
    """Reset only color parameters."""
    st.session_state.hue_shift = 0
    st.session_state.saturation_shift = 0
    st.session_state.brightness_shift = 0
    st.session_state.preview_output = None


def reset_fit_params():
    """Reset only fit parameters."""
    st.session_state.fit_preset = 'Original'
    st.session_state.sleeve_shift = 0
    st.session_state.preview_output = None


def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🎨 VITON-HD Interactive Editor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Virtual Try-On with Real-Time Editing</div>', unsafe_allow_html=True)
    
    # Load pipeline and preprocessor
    with st.spinner("Loading models..."):
        pipeline = load_pipeline()
        preprocessor = load_preprocessor()
    
    # Get available images
    person_images, cloth_images = preprocessor.get_available_images(mode='test')
    
    # Sidebar controls
    with st.sidebar:
        st.header("📤 Select Images")
        
        person_name = st.selectbox(
            "Person Image",
            person_images,
            help="Select a person image from the test dataset"
        )
        
        cloth_name = st.selectbox(
            "Cloth Image",
            cloth_images,
            help="Select a clothing item from the test dataset"
        )
        
        st.divider()
        
        # Color editing controls
        st.header("🎨 Color Editing")
        
        hue_shift = st.slider(
            "Hue Shift",
            min_value=-180,
            max_value=180,
            value=st.session_state.get('hue_shift', 0),
            step=5,
            help="Adjust the color hue (-180° to 180°)",
            key='hue_shift'
        )
        
        saturation_shift = st.slider(
            "Saturation",
            min_value=-100,
            max_value=100,
            value=st.session_state.get('saturation_shift', 0),
            step=5,
            help="Adjust color intensity (-100% to 100%)",
            key='saturation_shift'
        )
        
        brightness_shift = st.slider(
            "Brightness",
            min_value=-50,
            max_value=50,
            value=st.session_state.get('brightness_shift', 0),
            step=5,
            help="Adjust brightness (-50% to 50%)",
            key='brightness_shift'
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Color", use_container_width=True):
                reset_color_params()
                st.rerun()
        
        st.divider()
        
        # Fit adjustment controls
        st.header("🧵 Fit Adjustment")
        
        fit_preset = st.selectbox(
            "Fit Preset",
            ['Original', 'Slim (-10%)', 'Fitted (-5%)', 'Relaxed (+5%)', 'Oversized (+10%)'],
            index=0,
            help="Select a predefined fit adjustment",
            key='fit_preset'
        )
        
        sleeve_shift = st.slider(
            "Sleeve Length",
            min_value=-20,
            max_value=20,
            value=st.session_state.get('sleeve_shift', 0),
            step=2,
            help="Adjust sleeve length (-20 to 20 pixels)",
            key='sleeve_shift'
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Fit", use_container_width=True):
                reset_fit_params()
                st.rerun()
        
        st.divider()
        
        # Main action buttons
        st.header("⚡ Generate")
        
        if st.button("🔄 Reset All", type="secondary", use_container_width=True):
            reset_all_params()
            st.rerun()
        
        if st.button("⚡ Quick Preview (256px)", type="primary", use_container_width=True):
            st.session_state.run_preview = True
        
        if st.button("🎯 Generate HD (1024px)", type="primary", use_container_width=True):
            st.session_state.run_hd = True
    
    # Main content area
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Column 1: Original images
    with col1:
        st.subheader("📷 Original Images")
        
        # Load and display person image
        person_img_path = f"./datasets/test/image/{person_name}"
        person_img = Image.open(person_img_path)
        st.image(person_img, caption=f"Person: {person_name}", use_container_width=True)
        
        # Load and display cloth image
        cloth_img_path = f"./datasets/test/cloth/{cloth_name}"
        cloth_img = Image.open(cloth_img_path)
        st.image(cloth_img, caption=f"Cloth: {cloth_name}", use_container_width=True)
        
        # Show current settings
        with st.expander("📊 Current Settings"):
            st.write(f"**Color:**")
            st.write(f"- Hue: {hue_shift}°")
            st.write(f"- Saturation: {saturation_shift}%")
            st.write(f"- Brightness: {brightness_shift}%")
            st.write(f"**Fit:**")
            st.write(f"- Preset: {fit_preset}")
            st.write(f"- Sleeve: {sleeve_shift}px")
    
    # Column 2: Preview
    with col2:
        st.subheader("👁️ Preview (256x192)")
        st.caption("Fast preview - 5-10 seconds")
        
        if st.session_state.get('run_preview', False):
            with st.spinner("Generating preview..."):
                start_time = time.time()
                
                # Load data if not cached or if selection changed
                if (st.session_state.person_data is None or 
                    st.session_state.person_data['img_name'] != person_name):
                    st.session_state.person_data = preprocessor.load_person_data(person_name)
                
                if (st.session_state.cloth_data is None or 
                    st.session_state.cloth_data['cloth_name'] != cloth_name):
                    st.session_state.cloth_data = preprocessor.load_cloth_data(cloth_name)
                
                # Set to low resolution for preview
                pipeline.set_resolution(256, 192)
                
                # Run pipeline
                output, intermediates = pipeline.run_full_pipeline(
                    st.session_state.person_data['img_agnostic'],
                    st.session_state.person_data['parse_agnostic'],
                    st.session_state.person_data['pose'],
                    st.session_state.cloth_data['cloth'],
                    st.session_state.cloth_data['cloth_mask'],
                    return_intermediates=True
                )
                
                # Apply color edits
                if hue_shift != 0 or saturation_shift != 0 or brightness_shift != 0:
                    edited_warped_c = ColorEditor.adjust_hsv(
                        intermediates['warped_c'],
                        hue_shift=hue_shift,
                        saturation_shift=saturation_shift,
                        brightness_shift=brightness_shift
                    )
                else:
                    edited_warped_c = intermediates['warped_c']
                
                # Apply fit edits
                if fit_preset != 'Original':
                    edited_cm = MaskMorpher.apply_fit_preset(
                        intermediates['warped_cm'],
                        fit_preset
                    )
                else:
                    edited_cm = intermediates['warped_cm']
                
                # Apply sleeve adjustment
                if sleeve_shift != 0:
                    adjusted_parse, edited_cm = MaskMorpher.adjust_sleeve_length(
                        intermediates['parse'],
                        edited_cm,
                        sleeve_shift
                    )
                else:
                    adjusted_parse = intermediates['parse']
                
                # Regenerate if edited
                if (hue_shift != 0 or saturation_shift != 0 or brightness_shift != 0 or 
                    fit_preset != 'Original' or sleeve_shift != 0):
                    output = pipeline.run_alias(
                        intermediates['img_agnostic'],
                        intermediates['pose'],
                        edited_warped_c,
                        adjusted_parse,
                        edited_cm
                    )
                
                # Store intermediates for HD render
                st.session_state.intermediates = intermediates
                st.session_state.preview_output = output
                
                elapsed_time = time.time() - start_time
                st.success(f"✓ Preview generated in {elapsed_time:.1f}s")
            
            st.session_state.run_preview = False
        
        # Display preview
        if st.session_state.preview_output is not None:
            preview_img = tensor_to_pil(st.session_state.preview_output)
            st.image(preview_img, caption="Preview Output", use_container_width=True)
            
            # Download button for preview
            preview_bytes = pil_to_bytes(preview_img)
            st.download_button(
                label="💾 Download Preview",
                data=preview_bytes,
                file_name=f"preview_{person_name.split('.')[0]}_{cloth_name}",
                mime="image/jpeg",
                use_container_width=True
            )
        else:
            st.info("Click '⚡ Quick Preview' to generate a fast preview")
    
    # Column 3: HD Output
    with col3:
        st.subheader("🎯 Final HD (1024x768)")
        st.caption("High quality - 60-90 seconds")
        
        if st.session_state.get('run_hd', False):
            with st.spinner("Generating HD output... This may take 60-90 seconds"):
                start_time = time.time()
                
                # Load data if not cached
                if (st.session_state.person_data is None or 
                    st.session_state.person_data['img_name'] != person_name):
                    st.session_state.person_data = preprocessor.load_person_data(person_name)
                
                if (st.session_state.cloth_data is None or 
                    st.session_state.cloth_data['cloth_name'] != cloth_name):
                    st.session_state.cloth_data = preprocessor.load_cloth_data(cloth_name)
                
                # Set to full resolution
                pipeline.set_resolution(1024, 768)
                
                # Run pipeline
                output, intermediates = pipeline.run_full_pipeline(
                    st.session_state.person_data['img_agnostic'],
                    st.session_state.person_data['parse_agnostic'],
                    st.session_state.person_data['pose'],
                    st.session_state.cloth_data['cloth'],
                    st.session_state.cloth_data['cloth_mask'],
                    return_intermediates=True
                )
                
                # Apply same edits as preview
                if hue_shift != 0 or saturation_shift != 0 or brightness_shift != 0:
                    edited_warped_c = ColorEditor.adjust_hsv(
                        intermediates['warped_c'],
                        hue_shift=hue_shift,
                        saturation_shift=saturation_shift,
                        brightness_shift=brightness_shift
                    )
                else:
                    edited_warped_c = intermediates['warped_c']
                
                if fit_preset != 'Original':
                    edited_cm = MaskMorpher.apply_fit_preset(
                        intermediates['warped_cm'],
                        fit_preset
                    )
                else:
                    edited_cm = intermediates['warped_cm']
                
                if sleeve_shift != 0:
                    adjusted_parse, edited_cm = MaskMorpher.adjust_sleeve_length(
                        intermediates['parse'],
                        edited_cm,
                        sleeve_shift
                    )
                else:
                    adjusted_parse = intermediates['parse']
                
                # Regenerate if edited
                if (hue_shift != 0 or saturation_shift != 0 or brightness_shift != 0 or 
                    fit_preset != 'Original' or sleeve_shift != 0):
                    output = pipeline.run_alias(
                        intermediates['img_agnostic'],
                        intermediates['pose'],
                        edited_warped_c,
                        adjusted_parse,
                        edited_cm
                    )
                
                st.session_state.hd_output = output
                
                elapsed_time = time.time() - start_time
                st.success(f"✓ HD output generated in {elapsed_time:.1f}s")
            
            st.session_state.run_hd = False
        
        # Display HD output
        if st.session_state.hd_output is not None:
            hd_img = tensor_to_pil(st.session_state.hd_output)
            st.image(hd_img, caption="HD Output", use_container_width=True)
            
            # Download button for HD
            hd_bytes = pil_to_bytes(hd_img)
            st.download_button(
                label="💾 Download HD",
                data=hd_bytes,
                file_name=f"hd_{person_name.split('.')[0]}_{cloth_name}",
                mime="image/jpeg",
                use_container_width=True,
                type="primary"
            )
        else:
            st.info("Click '🎯 Generate HD' to create high-resolution output")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>VITON-HD Interactive Editor | Built with Streamlit | CPU-Compatible</p>
        <p><small>Editing tools: Color adjustment, Fit morphing, Zero retraining required</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
