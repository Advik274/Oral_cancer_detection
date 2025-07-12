import os
import streamlit as st
from tensorflow.keras.models import load_model
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

# Define a function to show model summaries
def show_summary():
    # Local model file paths
    model_paths = {
        'CNN': 'cnn_oral_cancer_model.keras',
        'ResNet50': 'resnet50_oral_cancer_model.keras',
        'EfficientNet': 'efficientNetB2_oral_cancer_model.keras',
        'VGG19': 'vgg19_oral_cancer_model.keras',
        'DenseNet': 'densenet121_oral_cancer_model.keras'
    }

    st.title("Model Summary Page")

    # User selects a model
    selected_model_name = st.selectbox(
        "Choose a model to summarize", list(model_paths.keys()))

    # Load the selected model when the button is clicked
    if st.button("Summarize Model"):
        model_path = model_paths[selected_model_name]

        # Check if the model file exists
        download_model_if_needed(selected_model_name, model_path)
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return

        # Load the model
        with st.spinner("Loading model..."):
            st.toast("Loading model...")
            model = load_model(model_path)

        st.success(f"Showing summary for {selected_model_name} model:")

        # Capture the summary in a list and display it
        summary_str = []
        model.summary(print_fn=lambda x: summary_str.append(x))
        st.text("\n".join(summary_str))

        st.toast("Summary created successfully!")

# Main function to display the page
def main():
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ["Model Summary", "Model Classification"])

    if options == "Model Summary":
        show_summary()
    elif options == "Model Classification":
        st.write("Model classification page placeholder.")  # You can hook this to your classify function.

if __name__ == "__main__":
    main()
