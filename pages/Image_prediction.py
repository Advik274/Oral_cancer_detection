import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import io
import zipfile
from PIL import Image
import matplotlib.pyplot as plt
import json
import random
import os
import base64
import gdown

# Define the model paths and their target sizes
model_paths = {
    'CNN': {
        'path': 'cnn_oral_cancer_model.keras',
        'target_size': (128, 128)
    },
    'ResNet50': {
        'path': 'resnet50_oral_cancer_model.keras',
        'target_size': (260, 260)
    },
    'EfficientNet': {
        'path': 'efficientNetB2_oral_cancer_model.keras',
        'target_size': (260, 260)
    },
    'DenseNet': {
        'path': 'densenet121_oral_cancer_model.keras',
        'target_size': (224, 224)
    },
    'VGG19': {
        'path': 'vgg19_oral_cancer_model.keras',
        'target_size': (224, 224)
    },
}

# Initialize session state variables
if 'saved_predictions' not in st.session_state:
    st.session_state.saved_predictions = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []

def load_existing_predictions():
    if os.path.exists('prediction_history.json'):
        with open('prediction_history.json', 'r') as f:
            return json.load(f)
    return []

# Load existing predictions into session state
st.session_state.saved_predictions = load_existing_predictions()

def save_predictions_to_history(uploaded_files, predictions, model_name):
    prediction_data = []
    for i, uploaded_file in enumerate(uploaded_files):
        actual = 'Cancer' if predictions[i][0] == 0 else 'Non Cancer'
        prediction_data.append({
            'file_name': uploaded_file.name,
            'model_used': model_name,
            'prediction': actual
        })

    st.session_state.saved_predictions.extend(prediction_data)

    with open('prediction_history.json', 'w') as f:
        json.dump(st.session_state.saved_predictions, f, indent=4)
    st.success("Predictions saved to history successfully.")

cancer_warning_messages = [
    "Please consult a doctor immediately.",
    "We recommend scheduling a medical check-up soon.",
    "It's crucial to seek medical advice right away.",
    "Contact your healthcare provider for further examination.",
    "This result may be concerning. Please consult a specialist."
]

def load_uploaded_images(uploaded_files, target_size):
    images = []
    for uploaded_file in uploaded_files:  # Removed enumerate which was causing the error
        try:
            image = load_img(uploaded_file, target_size=target_size)
            image_array = img_to_array(image)
            images.append(image_array)
        except Exception as e:
            st.error(f"Error loading image {uploaded_file.name}: {str(e)}")
    return np.array(images)

def show_image_prediction():
    # Streamlit UI
    st.title('Oral Cancer Detection Model Evaluation')

    # Model selection
    model_selection = st.selectbox("Select a model", list(model_paths.keys()))

    # Upload images
    uploaded_files = st.file_uploader(
        "Upload images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files:
        target_size = model_paths[model_selection]['target_size']
        X_test = load_uploaded_images(uploaded_files, target_size)

        # Function to evaluate the model on uploaded images
        def evaluate_model(model, images):
            predictions = model.predict(images)
            return (predictions > 0.5).astype(int)

        # Add a button to trigger predictions
        if st.button('Predict'):
            st.info("Loading the model...")
            
            model_path = model_paths[model_selection]['path']
            try:
                with st.spinner("Loading model..."):
                    download_model_if_needed(model_selection, model_path)
                    model_to_use = load_model(model_path)

                with st.spinner("Evaluating images..."):
                    st.session_state.predictions = evaluate_model(model_to_use, X_test)
                    st.session_state.uploaded_images = uploaded_files

                st.toast("✨ Images predicted successfully!")

                # Display predictions
                st.subheader('Predictions:')
                for i, uploaded_file in enumerate(uploaded_files):
                    # Handle case where there might be fewer predictions than images
                    if i < len(st.session_state.predictions):
                        actual = 'Cancer' if st.session_state.predictions[i][0] == 0 else 'Non Cancer'
                        caption = f'Predicted: {actual}'
                        st.image(uploaded_file, caption=caption, use_column_width=True)

                        if actual == 'Cancer':
                            warning_message = random.choice(cancer_warning_messages)
                            st.warning(warning_message)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.error(f"Please ensure the model file exists at: {model_path}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button('Clear'):
            st.session_state.predictions = []
            st.session_state.uploaded_images = []
            st.success("🗑️ Cleared all predictions and uploaded images.")

    with col2:
        if len(st.session_state.predictions) > 0 and len(st.session_state.uploaded_images) > 0:
            if st.button('Save Predictions'):
                save_predictions_to_history(
                    st.session_state.uploaded_images, st.session_state.predictions, model_selection)

    # Download predictions functionality
    if len(st.session_state.predictions) > 0 and len(st.session_state.uploaded_images) > 0:
        prediction_images = []
        for i, uploaded_file in enumerate(st.session_state.uploaded_images):
            if i < len(st.session_state.predictions):  # Safety check
                actual = 'Cancer' if st.session_state.predictions[i][0] == 0 else 'Non Cancer'
                image = Image.open(uploaded_file)
                pred_image = image.copy()
                plt.imshow(pred_image)
                plt.axis('off')
                plt.title(f'Predicted: {actual}')
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                plt.close()
                buf.seek(0)
                prediction_images.append((buf, f'prediction_{i + 1}.png'))

        if prediction_images:  # Only create zip if we have predictions
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zf:
                for image_buf, filename in prediction_images:
                    zf.writestr(filename, image_buf.getvalue())
            zip_buffer.seek(0)

            st.download_button(
                label="📥 Download Predictions",
                data=zip_buffer,
                file_name='predictions.zip',
                mime='application/zip'
            )

    # Utility function to convert an image to base64 for display
    def image_to_base64(image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    # Display logo
    logo_path = "./assets/logo.png"
    if os.path.exists(logo_path):
        try:
            logo_image = Image.open(logo_path)
            logo_base64 = image_to_base64(logo_image)
            st.sidebar.markdown(
                f"""
                <img src="data:image/jpeg;base64,{logo_base64}"
                    style="border-radius: 30px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); width: 90%; height: auto;" />
                """, unsafe_allow_html=True
            )
        except Exception as e:
            st.sidebar.warning(f"Could not load logo: {e}")

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

if __name__ == "__main__":
    show_image_prediction()