import streamlit as st
from streamlit_option_menu import option_menu
import json
import base64
import io
# Function to display model comparison
from PIL import Image


def show_model_comparison():

    selected_page = option_menu(
        menu_title=None,
        options=["Comparison", "Classification", "Summary"],
        icons=['bar-chart', 'check-square', 'book'],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )

    def display_page(page_name):
        if page_name == "Comparison":
            import comparison_pages.comparison as comparison  # Import model comparison module
            # Call function to display model comparison
            comparison.show_comparison()
        elif page_name == "Classification":
            import comparison_pages.classification as classification
            classification.show_classify()
        elif page_name == "Summary":
            import comparison_pages.summary as summary
            summary.show_summary()
    display_page(selected_page)

    def image_to_base64(image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    logo_path = "./assets/logo.png"  # Update with your logo file path
    logo_image = Image.open(logo_path)

    # Convert the logo image to base64
    logo_base64 = image_to_base64(logo_image)

    # Display the logo with custom CSS styles
    st.sidebar.markdown(
        f"""
        <img src="data:image/jpeg;base64,{logo_base64}"
            style="border-radius: 30px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); width: 90%; height: auto;" />
        """, unsafe_allow_html=True
    )
