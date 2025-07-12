from PIL import Image
import streamlit as st
from streamlit_lottie import st_lottie
import json
import base64
import io


def show_home_content():
    st.write("# Welcome to the OralGuard🦷")

    def load_lottie_file(filepath: str):
        with open(filepath, 'r', encoding='utf-8') as file:  # Specify UTF-8 encoding
            return json.load(file)

    # Load the first Lottie animation
    # Update with your first Lottie file path
    lottie_file_path_1 = "./assets/lottie.json"
    lottie_animation_1 = load_lottie_file(lottie_file_path_1)

    # Display the first Lottie animation
    st_lottie(
        lottie_animation_1,
        speed=1,
        loop=True,
        height=None,  # Adjust this height as needed
        width=None,  # Use None for automatic width
        quality="low",
        key='Oral Cancer Detection 1'
    )
    # Description about the project
    st.markdown(
        """
    Oral cancer is one of the most prevalent cancers worldwide, and early detection can significantly improve treatment outcomes.
    Our system leverages deep learning models and image processing techniques to detect early signs of oral cancer from medical images. 
    Using advanced convolutional neural networks (CNNs), we can identify patterns and anomalies in oral tissues, providing healthcare professionals with a tool to support early diagnosis.
    
    ### Key Features:
    - Real-time analysis of oral cavity images for abnormal tissue detection
    - High accuracy in predicting potential cancerous regions
    - Easy integration with medical imaging devices
    - Secure and privacy-focused data handling

    By utilizing this technology, we aim to assist in the early detection of oral cancer, improving patient outcomes and survival rates.

    You can find more details and the entire project on my [repo](https://github.com/advik274).
    """
    )
    # Load the second Lottie animation
    # # Update with your second Lottie file path
    lottie_file_path_2 = "./assets/lottie2.json"
    lottie_animation_2 = load_lottie_file(lottie_file_path_2)
    # Display the second Lottie animation
    st_lottie(
        lottie_animation_2,
        speed=1,
        loop=True,
        height=None,  # Adjust this height as needed
        width=None,  # Use None for automatic width
        quality="low",
        key='Oral Cancer Detection 2'
    )

    def image_to_base64(image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
    logo_path = "./assets/logo.jpg"  # Update with your logo file path
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
