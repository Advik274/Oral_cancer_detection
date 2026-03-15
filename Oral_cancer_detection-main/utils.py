import streamlit as st
import base64
import io
from PIL import Image
import os

def image_to_base64(image: Image.Image, format="PNG") -> str:
    """Convert a PIL image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()

def display_sidebar_logo(logo_path="./assets/logo.png"):
    """Display the logo in the sidebar with custom CSS."""
    if os.path.exists(logo_path):
        try:
            logo_image = Image.open(logo_path)
            # Use PNG for high quality, unless it's a JPG
            ext = os.path.splitext(logo_path)[1].lower()
            img_format = "JPEG" if ext in ['.jpg', '.jpeg'] else "PNG"
            
            logo_base64 = image_to_base64(logo_image, format=img_format)
            mime_type = "image/jpeg" if img_format == "JPEG" else "image/png"
            
            st.sidebar.markdown(
                f"""
                <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                    <img src="data:{mime_type};base64,{logo_base64}"
                        style="border-radius: 20px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3); width: 85%; height: auto;" />
                </div>
                """, unsafe_allow_html=True
            )
        except Exception as e:
            st.sidebar.warning(f"Could not load logo: {e}")
    else:
        st.sidebar.warning("Logo file not found.")

def safe_set_logo(logo_path="./assets/logo1.png"):
    """Safely call st.logo for newer Streamlit versions."""
    if hasattr(st, "logo"):
        try:
            st.logo(logo_path, size="large")
        except Exception:
            pass
