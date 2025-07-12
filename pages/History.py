import streamlit as st
import json
import os
import pandas as pd
import base64
import io
from PIL import Image
# Function to display history


def show_history():
    st.title("History")
    st.write("This section shows the history of predictions and analyses.")

    # Check if the history JSON file exists
    history_file = 'prediction_history.json'

    if os.path.exists(history_file):
        # Load the history data from the JSON file
        with open(history_file, 'r') as file:
            prediction_history = json.load(file)

        if len(prediction_history) > 0:
            # Display raw JSON data
            st.subheader("Raw JSON Data")
            st.json(prediction_history)

            # Convert JSON to DataFrame for tabular display
            st.subheader("DataFrame View")
            df = pd.DataFrame(prediction_history)
            st.dataframe(df)

        else:
            st.write("No prediction history found.")
    else:
        st.write(
            "Prediction history file does not exist yet. Save predictions first.")

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
