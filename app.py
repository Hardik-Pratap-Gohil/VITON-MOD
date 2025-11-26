"""
Streamlit app for VITON-HD with realistic cloth editing.
Focuses on colors, patterns, logos, and textures.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import os

from inference_pipeline import VITONInference
from preprocessing import VITONPreprocessor
from cloth_editor import ClothColorEditor, PatternGenerator, LogoApplicator, FabricSimulator


# Page config
st.set_page_config(
    page_title="VITON-HD Interactive Editor",
    page_icon="👕",
    layout="wide"
)

st.title("👕 VITON-HD Interactive Cloth Editor")
st.markdown("Edit clothing with colors, patterns, logos, and textures - then see it on a person!")


def tensor_to_pil(tensor):
    """Convert tensor to PIL Image for display."""
    img = tensor.squeeze(0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


@st.cache_resource
def load_pipeline():
    """Load VITON-HD models (cached)."""
    pipeline = VITONInference()
    pipeline.load_models(
        seg_checkpoint='seg_final.pth',
        gmm_checkpoint='gmm_final.pth',
        alias_checkpoint='alias_final.pth'
    )
    return pipeline


@st.cache_resource
def load_preprocessor():
    """Load preprocessor (cached)."""
    return VITONPreprocessor('./datasets')


def get_available_items():
    """Get lists of available persons and clothes."""
    person_dir = './datasets/test/image'
    cloth_dir = './datasets/test/cloth'
    
    persons = sorted([f for f in os.listdir(person_dir) if f.endswith('.jpg')])
    clothes = sorted([f for f in os.listdir(cloth_dir) if f.endswith('.jpg')])
    
    return persons, clothes


def main():
    # Load resources
    try:
        pipeline = load_pipeline()
        preprocessor = load_preprocessor()
        persons, clothes = get_available_items()
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return
    
    # Sidebar - Selection
    st.sidebar.header("📋 Selection")
    
    selected_person = st.sidebar.selectbox(
        "Choose Person",
        persons,
        index=0
    )
    
    selected_cloth = st.sidebar.selectbox(
        "Choose Cloth",
        clothes,
        index=0
    )
    
    resolution = st.sidebar.radio(
        "Resolution",
        ["Preview (256x192)", "HD (768x1024)"],
        index=1
    )
    
    # Set resolution
    if "Preview" in resolution:
        width, height = 192, 256
    else:
        width, height = 768, 1024
    
    pipeline.set_resolution(width=width, height=height)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Tip:** Edit the cloth, then click Generate to see results!")
    
    # Main area - Editing options
    st.header("✏️ Edit Cloth")
    
    # Create tabs for different editing categories
    tab1, tab2, tab3, tab4 = st.tabs(["🎨 Colors", "🎭 Patterns", "🏷️ Logo", "🧵 Texture"])
    
    # Store all edit parameters
    edits = {
        'color_preset': None,
        'hue': 0,
        'saturation': 0,
        'brightness': 0,
        'pattern_type': 'None',
        'pattern_params': {},
        'logo_text': '',
        'logo_params': {},
        'texture_type': 'None'
    }
    
    with tab1:
        st.subheader("Color Adjustments")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Preset Palettes**")
            edits['color_preset'] = st.radio(
                "Choose a palette",
                ['None', 'Vibrant', 'Pastel', 'Earth', 'Monochrome', 'Warm', 'Cool'],
                index=0
            )
        
        with col2:
            st.markdown("**Custom Adjustments**")
            edits['hue'] = st.slider("Hue Shift", -180, 180, 0, 5)
            edits['saturation'] = st.slider("Saturation", -100, 100, 0, 5)
            edits['brightness'] = st.slider("Brightness", -100, 100, 0, 5)
    
    with tab2:
        st.subheader("Pattern Overlay")
        
        edits['pattern_type'] = st.selectbox(
            "Pattern Type",
            ['None', 'Vertical Stripes', 'Horizontal Stripes', 'Polkadots', 'Checkerboard']
        )
        
        if edits['pattern_type'] != 'None':
            col1, col2 = st.columns(2)
            with col1:
                edits['pattern_params']['blend_mode'] = st.selectbox(
                    "Blend Mode",
                    ['multiply', 'overlay', 'screen'],
                    index=0
                )
                edits['pattern_params']['opacity'] = st.slider(
                    "Pattern Opacity",
                    0.0, 1.0, 0.6, 0.05
                )
            
            with col2:
                if 'Stripes' in edits['pattern_type']:
                    edits['pattern_params']['stripe_width'] = st.slider(
                        "Stripe Width (px)",
                        5, 50, 15, 1
                    )
                    edits['pattern_params']['color1'] = st.color_picker(
                        "Color 1", "#6666FF"
                    )
                    edits['pattern_params']['color2'] = st.color_picker(
                        "Color 2", "#FFFFFF"
                    )
                
                elif edits['pattern_type'] == 'Polkadots':
                    edits['pattern_params']['dot_radius'] = st.slider(
                        "Dot Radius (px)",
                        3, 20, 8, 1
                    )
                    edits['pattern_params']['spacing'] = st.slider(
                        "Dot Spacing (px)",
                        15, 60, 35, 5
                    )
                    edits['pattern_params']['bg_color'] = st.color_picker(
                        "Background", "#FFFFFF"
                    )
                    edits['pattern_params']['dot_color'] = st.color_picker(
                        "Dot Color", "#323232"
                    )
                
                elif edits['pattern_type'] == 'Checkerboard':
                    edits['pattern_params']['square_size'] = st.slider(
                        "Square Size (px)",
                        10, 50, 25, 5
                    )
                    edits['pattern_params']['color1'] = st.color_picker(
                        "Color 1", "#C8C8C8"
                    )
                    edits['pattern_params']['color2'] = st.color_picker(
                        "Color 2", "#646464"
                    )
    
    with tab3:
        st.subheader("Add Text Logo")
        
        edits['logo_text'] = st.text_input(
            "Logo Text",
            placeholder="e.g., BRAND, NIKE, 23"
        )
        
        if edits['logo_text']:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                edits['logo_params']['position'] = st.selectbox(
                    "Position",
                    ['center', 'top', 'bottom'],
                    index=0
                )
            
            with col2:
                edits['logo_params']['font_size'] = st.slider(
                    "Font Size",
                    20, 120, 60, 5
                )
            
            with col3:
                logo_color = st.color_picker("Logo Color", "#FFFFFF")
                # Convert hex to RGB tuple
                edits['logo_params']['color'] = tuple(int(logo_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    with tab4:
        st.subheader("Fabric Texture")
        
        edits['texture_type'] = st.radio(
            "Texture Type",
            ['None', 'Canvas', 'Denim', 'Silk', 'Linen'],
            index=0
        )
    
    # Generate button
    st.markdown("---")
    generate_button = st.button("🎨 Generate Try-On", type="primary", use_container_width=True)
    
    if generate_button:
        with st.spinner("Loading data..."):
            try:
                # Load person and cloth data
                person_data = preprocessor.load_person_data(selected_person)
                cloth_data = preprocessor.load_cloth_data(selected_cloth)
                
                img_agnostic = person_data['img_agnostic']
                parse_agnostic = person_data['parse_agnostic']
                pose = person_data['pose']
                c = cloth_data['cloth']
                cm = cloth_data['cloth_mask']
                
            except Exception as e:
                st.error(f"Failed to load data: {str(e)}")
                return
        
        # Apply edits to cloth
        with st.spinner("Applying edits to cloth..."):
            try:
                c_edited = c.clone()
                
                # 1. Color Preset
                if edits['color_preset'] != 'None':
                    c_edited = ClothColorEditor.apply_color_palette(
                        c_edited, cm, edits['color_preset'].lower()
                    )
                
                # 2. Custom Color Adjustments
                if edits['hue'] != 0 or edits['saturation'] != 0 or edits['brightness'] != 0:
                    c_edited = ClothColorEditor.adjust_hsv(
                        c_edited, cm,
                        hue_shift=edits['hue'],
                        saturation_shift=edits['saturation'],
                        brightness_shift=edits['brightness']
                    )
                
                # 3. Pattern
                if edits['pattern_type'] != 'None':
                    _, _, h, w = c.shape
                    
                    # Helper to convert hex to RGB tuple
                    def hex_to_rgb(hex_color):
                        hex_color = hex_color.lstrip('#')
                        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    
                    if edits['pattern_type'] == 'Vertical Stripes':
                        pattern = PatternGenerator.create_stripes(
                            h, w, 'vertical',
                            edits['pattern_params']['stripe_width'],
                            hex_to_rgb(edits['pattern_params']['color1']),
                            hex_to_rgb(edits['pattern_params']['color2'])
                        )
                    elif edits['pattern_type'] == 'Horizontal Stripes':
                        pattern = PatternGenerator.create_stripes(
                            h, w, 'horizontal',
                            edits['pattern_params']['stripe_width'],
                            hex_to_rgb(edits['pattern_params']['color1']),
                            hex_to_rgb(edits['pattern_params']['color2'])
                        )
                    elif edits['pattern_type'] == 'Polkadots':
                        pattern = PatternGenerator.create_polkadots(
                            h, w,
                            edits['pattern_params']['dot_radius'],
                            edits['pattern_params']['spacing'],
                            hex_to_rgb(edits['pattern_params']['bg_color']),
                            hex_to_rgb(edits['pattern_params']['dot_color'])
                        )
                    elif edits['pattern_type'] == 'Checkerboard':
                        pattern = PatternGenerator.create_checkerboard(
                            h, w,
                            edits['pattern_params']['square_size'],
                            hex_to_rgb(edits['pattern_params']['color1']),
                            hex_to_rgb(edits['pattern_params']['color2'])
                        )
                    
                    c_edited = PatternGenerator.apply_pattern(
                        c_edited, cm, pattern,
                        edits['pattern_params']['blend_mode'],
                        edits['pattern_params']['opacity']
                    )
                
                # 4. Logo
                if edits['logo_text']:
                    c_edited = LogoApplicator.add_text_logo(
                        c_edited, cm,
                        text=edits['logo_text'],
                        position=edits['logo_params']['position'],
                        font_size=edits['logo_params']['font_size'],
                        color=edits['logo_params']['color']
                    )
                
                # 5. Texture
                if edits['texture_type'] != 'None':
                    c_edited = FabricSimulator.add_subtle_texture(
                        c_edited, cm, edits['texture_type'].lower()
                    )
                
            except Exception as e:
                st.error(f"Failed to apply edits: {str(e)}")
                st.exception(e)
                return
        
        # Run VITON pipeline
        with st.spinner(f"Running VITON-HD pipeline ({width}x{height})..."):
            try:
                output, _ = pipeline.run_full_pipeline(
                    img_agnostic, parse_agnostic, pose, c_edited, cm,
                    return_intermediates=True
                )
            except Exception as e:
                st.error(f"Pipeline failed: {str(e)}")
                st.exception(e)
                return
        
        # Display results
        st.success("✅ Generation complete!")
        
        st.header("📊 Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original Cloth")
            st.image(tensor_to_pil(c), use_container_width=True)
        
        with col2:
            st.subheader("Edited Cloth")
            st.image(tensor_to_pil(c_edited), use_container_width=True)
        
        with col3:
            st.subheader("Final Try-On")
            st.image(tensor_to_pil(output), use_container_width=True)
        
        # Download button
        st.markdown("---")
        output_img = tensor_to_pil(output)
        
        import io
        buf = io.BytesIO()
        output_img.save(buf, format='JPEG')
        
        st.download_button(
            label="⬇️ Download Result",
            data=buf.getvalue(),
            file_name=f"viton_{selected_person.replace('.jpg', '')}_{selected_cloth.replace('.jpg', '')}.jpg",
            mime="image/jpeg"
        )


if __name__ == "__main__":
    main()
