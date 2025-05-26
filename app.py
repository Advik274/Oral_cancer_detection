from streamlit_option_menu import option_menu
from PIL import Image
import streamlit as st
from streamlit_lottie import st_lottie
import json
import base64
import io

# Set the page configuration with a tongue or mouth emoji as the page icon
st.set_page_config(page_title="Oral Cancer Detection",
                   page_icon="🦷", layout="wide")

# Load and display logo
st.logo("./assets/logo1.png", size="large", link=None, icon_image=None)

st.sidebar.title("Oral Cancer Detection")

# Add CSS for the spinner
st.markdown(
    """
    <style>
    .streamlit-spinner {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;  /* Full height for centering */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to display the selected page
def display_page(page_name):
    with st.spinner("Loading..."):
        if page_name == "Home":
            import pages.Home as home  # Import the home module
            home.show_home_content()    # Call the function to display home content
        elif page_name == "About":
            import pages.About as about  # Import model comparison module
            # Call function to display model comparison
            about.show_about()
        elif page_name == "Model Comparison":
            import pages.Model_comparison as model_comparison  # Import model comparison module
            # Call function to display model comparison
            model_comparison.show_model_comparison()
        elif page_name == "Image Prediction":
            import pages.Image_prediction as image_prediction  # Import image prediction module
            # Call function to display image prediction
            image_prediction.show_image_prediction()
        elif page_name == "Real-Time Detection":
            # Import real-time detection module
            import pages.Real_time_detection as real_time_detection
            # Call function to display real-time detection
            real_time_detection.show_real_time_detection()
        elif page_name == "History":
            import pages.History as history  # Import history module
            history.show_history()            # Call function to display history

# Sidebar for navigation using option_menu
with st.sidebar:
    selected_page = option_menu(
        menu_title=None,  # Required
        options=["Home", "About",  "Model Comparison", "Image Prediction",
                 "Real-Time Detection", "History"],  # Pages
        icons=["house", "info-circle", "list-task", "image", "camera-video",
               "clock-history"],  # Icons for each page
        menu_icon="cast",  # Main menu icon
        default_index=0,  # Default page (Home)
        orientation="vertical",  # Keep the menu vertical
    )

# Call the function to display the selected page
display_page(selected_page)
