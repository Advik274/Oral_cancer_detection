import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import streamlit as st
import gdown

# Define local model paths and target sizes
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

# Hardcoded Google Drive test folder URL
TEST_FOLDER_URL = "https://drive.google.com/drive/folders/1M2w4lWrV5iJ3lML78tqv2MLIij8Q89U8?usp=sharing"

def download_test_folder_from_drive(output_dir="test"):
    import gdown
    folder_id = TEST_FOLDER_URL.split("folders/")[1].split("?")[0]
    gdown.download_folder(id=folder_id, output=output_dir, quiet=False, use_cookies=False)

def show_comparison():
    # Load images and labels
    def load_data(folder_path, target_size):
        images = []
        labels = []
        for label, category in enumerate(['CANCER', 'NON CANCER']):
            category_folder = os.path.join(folder_path, category)
            for file_name in os.listdir(category_folder):
                image_path = os.path.join(category_folder, file_name)
                image = load_img(image_path, target_size=target_size)
                image_array = img_to_array(image)
                images.append(image_array)
                labels.append(label)
        return np.array(images), np.array(labels)

    # Evaluate a given model
    def evaluate_model(model, test_path, target_size):
        images, labels = load_data(test_path, target_size)
        images = images / 255.0  # Normalize
        predictions = model.predict(images)
        predicted_classes = (predictions > 0.5).astype(int)

        accuracy = np.mean(predicted_classes.flatten() == labels.flatten()) * 100

        # Probability stats
        total_cancer_prob = np.sum(predictions)
        total_non_cancer_prob = np.sum(1 - predictions)
        num_images = len(predictions)

        avg_cancer_prob = (total_cancer_prob / num_images) * 100
        avg_non_cancer_prob = (total_non_cancer_prob / num_images) * 100

        return accuracy, avg_cancer_prob, avg_non_cancer_prob

    # Load selected model from local path
    def load_selected_model(model_name):
        model_info = model_paths.get(model_name)
        if model_info:
            model_path = model_info['path']
            download_model_if_needed(model_name, model_path)
            if os.path.exists(model_path):
                with st.spinner(f"Loading {model_name} model..."):
                    model = load_model(model_path)
                    st.toast(f"{model_name} model loaded.")
                return model, model_info['target_size']
            else:
                st.error(f"Model file not found: {model_path}")
                return None, None
        else:
            st.error("Unknown model selected.")
            return None, None

    # Streamlit UI
    st.title("Model Comparison for Cancer Detection")
    st.write("Probability < 0.5 indicates Cancer; >= 0.5 indicates Non-Cancer.")

    selected_models = st.multiselect("Select Models to Compare", list(model_paths.keys()))

    if st.button("Download Test Data from Google Drive"):
        download_test_folder_from_drive()
        st.success("Test data downloaded!")

    # Use the downloaded folder for evaluation
    test_data_path = "./test"

    if st.button("Evaluate Selected Models"):
        results = []

        for model_name in selected_models:
            model, target_size = load_selected_model(model_name)
            if model:
                accuracy, cancer_prob, non_cancer_prob = evaluate_model(model, test_data_path, target_size)
                results.append({
                    'Model': model_name,
                    'Accuracy (%)': accuracy,
                    'Avg Cancer Probability (%)': cancer_prob,
                    'Avg Non Cancer Probability (%)': non_cancer_prob
                })

        if results:
            df = pd.DataFrame(results)
            st.toast("Model comparison complete.")
            st.write(df)

# Run the app
if __name__ == "__main__":
    show_comparison()
