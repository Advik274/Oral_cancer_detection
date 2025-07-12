import os
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import tempfile
import gdown

# Google Drive file IDs for each model
MODEL_DRIVE_IDS = {
    'CNN': '1W5Tenm8oFwqloXhohEdChuw3RkgfCg3q',
    'ResNet50': '1NX0EGwu7Wl335M-xImpuxHwYdIR_eLSB',
    'EfficientNet': '1HMp5FXECYzNeQQQy0oNnasWeJ70Il9bv',
    'DenseNet': '1l0OefSAn9mHZlSFyTD3R3FZrq3kYZI0J',
    'VGG19': '1UX0-Rvet8keEPxPE2Utth-oQPOB_pnPU',
}

def download_model_if_needed(model_name, local_path):
    file_id = MODEL_DRIVE_IDS.get(model_name)
    if file_id and not os.path.exists(local_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, local_path, quiet=False)

def show_plots():
    st.title("Neural Network Model Architecture Visualization")

    # Local model file paths
    model_paths = {
        'CNN': 'cnn_oral_cancer_model.keras',
        'ResNet50': 'resnet50_oral_cancer_model.keras',
        'EfficientNet': 'efficientNetB2_oral_cancer_model.keras',
        'VGG19': 'vgg19_oral_cancer_model.keras',
        'DenseNet': 'densenet121_oral_cancer_model.keras',
    }

    # User selects the model
    selected_model_name = st.selectbox("Choose a model", list(model_paths.keys()))

    # Button to plot the model architecture
    if st.button("Plot Model Architecture"):
        model_path = model_paths[selected_model_name]

        download_model_if_needed(selected_model_name, model_path)
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return

        # Load the selected model
        with st.spinner("Loading model..."):
            st.toast(f"Loading {selected_model_name} model...")
            model = load_model(model_path)

        # Create a temporary file to store the plot
        with st.spinner("Creating architecture plot..."):
            st.toast("Creating plot...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                plot_model(model, to_file=tmpfile.name, show_shapes=True,
                           show_layer_names=True, dpi=300)
                tmpfile.seek(0)
                img_data = tmpfile.read()

        # Display the plot
        st.image(img_data, caption=f"{selected_model_name} Architecture", use_column_width=True)

        # Download button
        st.download_button(
            label="Download Model Architecture Plot",
            data=img_data,
            file_name=f"{selected_model_name}_architecture.png",
            mime="image/png"
        )

if __name__ == "__main__":
    show_plots()
