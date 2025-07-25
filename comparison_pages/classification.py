import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import gdown
import zipfile

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

# Hardcoded Google Drive test zip file URL
TEST_ZIP_URL = "https://drive.google.com/uc?id=1dCJGxtebi9yJ-ZVeVb50UeATuUdhgxx3"
ZIP_PATH = "test.zip"
EXTRACT_DIR = "test"

def download_and_extract_test_zip():
    if not os.path.exists(EXTRACT_DIR):
        gdown.download(TEST_ZIP_URL, ZIP_PATH, quiet=False)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        # Debug: List the extracted files and folders
        import os
        for root, dirs, files in os.walk(EXTRACT_DIR):
            st.write(f"Extracted to: {root}")
            st.write(f"Subfolders: {dirs}")
            st.write(f"Files: {files}")

# Function to load images and labels
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

# Function to load model from local path
def load_local_model(model_path):
    # Find model name from path
    model_name = None
    for name, info in model_paths.items():
        if info['path'] == model_path:
            model_name = name
            break
    if model_name:
        download_model_if_needed(model_name, model_path)
    if os.path.exists(model_path):
        with st.spinner("Loading model..."):
            model = load_model(model_path)
            st.toast("Model loaded successfully!")
        return model
    else:
        st.error(f"Model file not found: {model_path}")
        return None

# Function to evaluate the selected model
def evaluate_model(model, test_data_path, target_size):
    # Load new unseen data
    X_test, y_test = load_data(test_data_path, target_size)

    # Normalize images
    X_test = X_test / 255.0

    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = (predictions > 0.5).astype(int)

    # Calculate accuracy
    accuracy = np.mean(predicted_classes.flatten() == y_test.flatten())
    st.write(f'Accuracy: {accuracy:.4f}')

    # Generate confusion matrix and classification report
    cm = confusion_matrix(y_test, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cancer", "Non Cancer"])

    # Plot confusion matrix
    st.subheader('Confusion Matrix')
    with st.spinner("Creating confusion matrix..."):
        fig, ax = plt.subplots()
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        st.pyplot(fig)
        st.toast("Confusion matrix created!")

    # Display classification report
    report = classification_report(y_test, predicted_classes, target_names=["Cancer", "Non Cancer"])
    st.subheader("Classification Report")
    st.text(report)
    st.toast("Classification report created!")

# Streamlit app structure
def show_classify():
    st.title("Cancer Detection Model Evaluation")
    st.write("Select a model and use the provided test dataset from Google Drive (zipped).")

    selected_model_name = st.selectbox("Choose a model", list(model_paths.keys()))

    if st.button("Download Test Data from Google Drive"):
        download_and_extract_test_zip()
        st.success("Test data downloaded and extracted!")

    # Use the downloaded folder for evaluation
    test_data_path = "./test"

    if st.button("Evaluate Model"):
        # Ensure test data is available
        download_and_extract_test_zip()
        # Check for correct folder structure
        if not os.path.exists(os.path.join(test_data_path, 'CANCER')) or not os.path.exists(os.path.join(test_data_path, 'NON CANCER')):
            st.error("Test data not found or folder structure is incorrect after extraction.")
            return
        if os.path.exists(test_data_path):
            model_details = model_paths[selected_model_name]
            model = load_local_model(model_details['path'])
            if model:
                evaluate_model(model, test_data_path, model_details['target_size'])
        else:
            st.error("The test data folder does not exist. Please download it first.")

# Main function to run the app
if __name__ == "__main__":
    show_classify()
