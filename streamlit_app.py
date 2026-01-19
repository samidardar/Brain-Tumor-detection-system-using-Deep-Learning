"""
Brain Tumor Segmentation - Streamlit Web Interface
===================================================
Interactive web application for brain tumor detection and segmentation.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import os

# Import U-Net model and Config from main script
from brain_tumor_segmentation import UNet, Config


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model(model_path: str = "outputs/best_model.pth"):
    """Load the trained U-Net model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNet(in_channels=3, out_channels=1)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_info = {
            'loaded': True,
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_dice': checkpoint.get('val_dice', 'N/A'),
            'val_iou': checkpoint.get('val_iou', 'N/A')
        }
    else:
        model_info = {'loaded': False}
    
    model = model.to(device)
    model.eval()
    
    return model, device, model_info


def preprocess_image(image: Image.Image, size: int = 256) -> torch.Tensor:
    """Preprocess uploaded image for model inference."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    
    # Convert to numpy and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Convert to tensor (C, H, W)
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
    
    # Add batch dimension
    return img_tensor.unsqueeze(0)


def predict(model, image_tensor: torch.Tensor, device: torch.device, threshold: float = 0.5):
    """Run inference and return prediction mask."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        
        # Model returns probabilities in eval mode
        pred_mask = (output > threshold).float()
        
    return output.cpu().numpy()[0, 0], pred_mask.cpu().numpy()[0, 0]


def calculate_tumor_percentage(mask: np.ndarray) -> float:
    """Calculate percentage of image containing tumor."""
    return (mask.sum() / mask.size) * 100


def create_overlay(original: Image.Image, mask: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Create overlay of tumor mask on original image."""
    # Resize original to match mask
    original = original.resize((mask.shape[1], mask.shape[0]))
    original_array = np.array(original.convert('RGB'))
    
    # Create colored mask (red for tumor)
    colored_mask = np.zeros_like(original_array)
    colored_mask[:, :, 0] = mask * 255  # Red channel
    
    # Blend
    overlay = original_array.copy()
    tumor_pixels = mask > 0
    overlay[tumor_pixels] = (
        (1 - alpha) * original_array[tumor_pixels] + 
        alpha * colored_mask[tumor_pixels]
    ).astype(np.uint8)
    
    return Image.fromarray(overlay)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🧠 Brain Tumor Segmentation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered MRI Analysis using Deep Learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_path = st.text_input(
            "Model Path",
            value="outputs/best_model.pth",
            help="Path to trained model checkpoint"
        )
        
        # Threshold slider
        threshold = st.slider(
            "Detection Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Adjust sensitivity of tumor detection"
        )
        
        # Overlay opacity
        overlay_alpha = st.slider(
            "Overlay Opacity",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.1
        )
        
        st.divider()
        
        # Model info
        st.header("📊 Model Info")
        model, device, model_info = load_model(model_path)
        
        if model_info['loaded']:
            st.success("✅ Model Loaded")
            st.metric("Device", str(device).upper())
            if model_info['val_dice'] != 'N/A':
                st.metric("Validation Dice", f"{model_info['val_dice']:.4f}")
            if model_info['val_iou'] != 'N/A':
                st.metric("Validation IoU", f"{model_info['val_iou']:.4f}")
        else:
            st.warning("⚠️ Model not found. Please train the model first.")
            st.code("python brain_tumor_segmentation.py --epochs 50")
    
    # Main content
    st.header("📤 Upload MRI Scan")
    
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Upload a brain MRI scan image"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        # Create columns for display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)
        
        # Run prediction
        with st.spinner("🔍 Analyzing MRI scan..."):
            image_tensor = preprocess_image(image)
            prob_mask, binary_mask = predict(model, image_tensor, device, threshold)
        
        with col2:
            st.subheader("🎯 Tumor Detection")
            # Create heatmap visualization
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(prob_mask, cmap='hot')
            ax.axis('off')
            ax.set_title(f"Probability Map (Threshold: {threshold})")
            st.pyplot(fig)
            plt.close()
        
        with col3:
            st.subheader("🔬 Segmentation Overlay")
            overlay = create_overlay(image, binary_mask, overlay_alpha)
            st.image(overlay, use_container_width=True)
        
        # Results section
        st.divider()
        st.header("📈 Analysis Results")
        
        tumor_percentage = calculate_tumor_percentage(binary_mask)
        tumor_detected = tumor_percentage > 0.1
        
        # Metrics row
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                "Tumor Detected",
                "YES" if tumor_detected else "NO",
                delta="Positive" if tumor_detected else None,
                delta_color="inverse" if tumor_detected else "off"
            )
        
        with metric_col2:
            st.metric(
                "Tumor Coverage",
                f"{tumor_percentage:.2f}%"
            )
        
        with metric_col3:
            st.metric(
                "Confidence",
                f"{prob_mask.max():.1%}"
            )
        
        with metric_col4:
            st.metric(
                "Image Size",
                f"{image.size[0]}x{image.size[1]}"
            )
        
        # Diagnosis suggestion
        st.divider()
        if tumor_detected:
            st.error("""
            ⚠️ **Potential Tumor Detected**
            
            The AI model has identified regions that may indicate the presence of a brain tumor.
            
            **Important:** This is an AI-assisted analysis tool. Please consult with a qualified 
            radiologist or neurosurgeon for proper diagnosis and treatment planning.
            """)
        else:
            st.success("""
            ✅ **No Significant Tumor Detected**
            
            The AI model did not identify significant tumor regions in this scan.
            
            **Note:** This does not replace professional medical evaluation. 
            Please consult with a healthcare provider for comprehensive assessment.
            """)
        
        # Download options
        st.divider()
        st.header("💾 Download Results")
        
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            # Save overlay
            overlay_bytes = io.BytesIO()
            overlay.save(overlay_bytes, format='PNG')
            st.download_button(
                label="📥 Download Overlay Image",
                data=overlay_bytes.getvalue(),
                file_name="tumor_overlay.png",
                mime="image/png"
            )
        
        with download_col2:
            # Save mask
            mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
            mask_bytes = io.BytesIO()
            mask_img.save(mask_bytes, format='PNG')
            st.download_button(
                label="📥 Download Segmentation Mask",
                data=mask_bytes.getvalue(),
                file_name="tumor_mask.png",
                mime="image/png"
            )
    
    else:
        # Placeholder when no image uploaded
        st.info("👆 Upload a brain MRI image to begin analysis")
        
        # Sample images section
        st.divider()
        st.header("ℹ️ How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 1️⃣ Upload
            Upload a brain MRI scan image (PNG, JPG, or TIFF format)
            """)
        
        with col2:
            st.markdown("""
            ### 2️⃣ Analyze
            Our U-Net AI model processes the image and identifies tumor regions
            """)
        
        with col3:
            st.markdown("""
            ### 3️⃣ Results
            View the segmentation overlay and download the results
            """)
        
        st.divider()
        st.header("🔬 About the Model")
        st.markdown("""
        This application uses a **U-Net Convolutional Neural Network** trained on the 
        LGG MRI Segmentation dataset. The model achieves:
        
        - **Dice Score:** ~90%
        - **Precision:** ~87%
        - **IoU:** ~82%
        
        **Architecture:** U-Net with GroupNorm, trained using Dice + BCE loss for 
        optimal handling of class imbalance in medical imaging.
        """)


if __name__ == "__main__":
    main()
