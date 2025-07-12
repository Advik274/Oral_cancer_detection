import os
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import base64
import io
import gdown

# Define local model paths with target sizes
model_info = {
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

# Predict a single image
def predict_image(model, image, target_size):
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return (prediction > 0.5).astype(int)

# Convert image to base64 for sidebar logo
def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Real-time detection function
def show_real_time_detection():
    st.title('Real-Time Oral Cancer Detection')

    selected_model_name = st.selectbox("Select a model", list(model_info.keys()))
    col1, col2 = st.columns(2)

    if 'capturing' not in st.session_state:
        st.session_state.capturing = False

    with col1:
        if st.button('Start Video'):
            st.session_state.capturing = True
            cap = cv2.VideoCapture(0)
            video_placeholder = st.empty()

            model_path = model_info[selected_model_name]['path']
            target_size = model_info[selected_model_name]['target_size']

            download_model_if_needed(selected_model_name, model_path)
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
                return

            with st.spinner("Loading model..."):
                model = load_model(model_path)
                st.toast(f"{selected_model_name} model loaded.")

            while st.session_state.capturing:
                ret, frame = cap.read()
                if not ret:
                    break

                prediction = predict_image(model, frame, target_size)
                result = 'Cancer' if prediction[0][0] == 0 else 'Non Cancer'

                cv2.putText(frame, f'Prediction: {result}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels='RGB', use_column_width=True)

            cap.release()

    with col2:
        if st.button('Stop Video'):
            st.session_state.capturing = False

    # Sidebar logo
    logo_path = "./assets/logo.png"
    if os.path.exists(logo_path):
        logo_image = Image.open(logo_path)
        logo_base64 = image_to_base64(logo_image)
        st.sidebar.markdown(
            f"""
            <img src="data:image/png;base64,{logo_base64}"
                style="border-radius: 30px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); width: 90%; height: auto;" />
            """, unsafe_allow_html=True
        )
    else:
        st.sidebar.warning("Logo not found.")

# Entry point
if __name__ == "__main__":
    show_real_time_detection()
