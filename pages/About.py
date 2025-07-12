
from PIL import Image
import streamlit as st
from streamlit_lottie import st_lottie
import json
import base64
import io
import pandas as pd


def show_about():
    st.title("About the Project")
    # 1. Introduction
    st.header("Introduction")
    st.write("""Oral cancer is a significant global health problem, with early detection being critical for effective treatment. Recent advancements in deep learning have paved the way for automated, accurate detection of diseases from medical images. This project utilizes state-of-the-art deep learning models to detect oral cancer, improving the efficiency and accuracy of diagnosis compared to traditional manual methods.""")
    # 2. Problem Statement
    st.header("Problem Statement")
    st.write("""Manual detection of oral cancer from medical images can be both time-consuming and error-prone, often leading to delayed diagnosis. This delay can result in the progression of the disease to more advanced stages, reducing the chances of successful treatment. There is a need for an automated system capable of quickly and accurately identifying early signs of oral cancer from medical images.""")
    # 3. Solution
    st.header("Solution")
    st.write("""This project aims to develop a deep learning-based solution for detecting oral cancer. The solution leverages multiple state-of-the-art neural network architectures such as CNN, ResNet50, DenseNet121, EfficientNetB2, and VGG19. These models are trained and evaluated to determine the best-performing architecture based on accuracy, speed, and memory efficiency.""")

    # 4. Methodologies
    st.header("Methodologies")
    # Create an expander for the flowchart section
    with st.expander("View Methodology Flowchart"):
        st.write(
            "The following flowchart illustrates the overall process of oral cancer detection used in this project:")
        flowchart_img = Image.open("./assets/flow.jpg")
        st.image(flowchart_img, caption="Methodology Flowchart")

    st.write("""The key steps involved are:
1. **Data Collection**: Medical images are gathered from a reliable source or dataset.
2. **Preprocessing**: The images are resized to a consistent shape (260x260x3), normalized, and augmented to ensure robustness during training.
3. **Model Selection**: Several deep learning models are selected for comparison: CNN, ResNet50, DenseNet121, EfficientNetB2, and VGG19.
4. **Training**: Each model is trained using the preprocessed data. The training process involves adjusting weights through backpropagation to minimize loss.
5. **Evaluation**: After training, the models are evaluated based on accuracy, speed, and memory usage. Various metrics such as accuracy, precision, recall, and F1 score are used for performance comparison.
6. **Comparison**: The models' performances are compared, and the best model is selected for deployment based on the use case requirements.
""")
    # 5. Models Used
    st.header("Models Used")
    st.write("""The following models were used in this project:
1. **Convolutional Neural Network (CNN)**: A basic but powerful architecture that excels in image classification tasks. CNNs are easy to train and perform well on simpler problems.
2. **ResNet50**: A deeper network that uses residual connections to mitigate the vanishing gradient problem, making it highly effective for complex tasks like medical image classification.
3. **DenseNet121**: Known for its dense connections, this model reuses features and efficiently captures information, leading to strong performance on tasks like cancer detection.
4. **EfficientNetB2**: This architecture scales model dimensions efficiently and is highly accurate while using fewer parameters, making it ideal for resource-constrained tasks.
5. **VGG19**: A very deep network with 19 layers, VGG19 is known for its simplicity and power in transfer learning. However, it is resource-intensive and slow.
""")

    # 6. About Models
    st.header("Model Comparison")
    # Create a DataFrame for model comparison
    data = {
        "Model": [
            "CNN",
            "ResNet50",
            "DenseNet121",
            "EfficientNetB2",
            "VGG19"
        ],
        "Parameters": [
            "Few",
            "Moderate",
            "Moderate",
            "Few",
            "Very High"
        ],
        "Accuracy": [
            "Moderate",
            "High",
            "High",
            "High",
            "High"
        ],
        "Speed": [
            "Fast",
            "Moderate",
            "Slower",
            "Fast",
            "Slow"
        ],
        "Memory Usage": [
            "Low",
            "Moderate",
            "High",
            "Low",
            "Very High"
        ],
        "Key Strengths": [
            "Simple and fast to train",
            "Deep network with residual connections",
            "Dense connections, better feature reuse",
            "Efficient scaling and accuracy",
            "Excellent for transfer learning, strong results"
        ],
        "Ideal Use Case": [
            "Basic image classification, initial tasks",
            "Complex medical image classification",
            "Memory-efficient deep learning tasks",
            "Resource-constrained, accuracy-critical tasks",
            "Large-scale tasks when memory isn’t a concern"
        ]
    }

    # Create the DataFrame
    model_comparison_df = pd.DataFrame(data)

    # Display the DataFrame as a table in the Streamlit app
    st.write("Here’s a detailed comparison of the models used in this project:")
    st.dataframe(model_comparison_df)

    # 7. Results
    st.header("Results - Model Comparison & Differences")
    st.write("""### Key Findings:
- **CNN**: Fast to train but provides only moderate accuracy, making it more suitable for initial experiments or simpler tasks.
- **ResNet50**: Achieves high accuracy with moderate speed, making it an excellent choice for complex tasks like medical image detection.
- **DenseNet121**: Delivers high accuracy with slower performance, but its dense connections make it memory efficient.
- **EfficientNetB2**: Offers the best balance between speed, accuracy, and low memory usage, making it ideal for tasks where resources are constrained.
- **VGG19**: Although powerful, VGG19’s very high memory usage and slow speed make it less practical for real-time applications but useful in transfer learning scenarios.
""")

    # 8. Conclusion
    st.header("Conclusion")
    st.write("""Deep learning models can significantly improve the accuracy and efficiency of oral cancer detection from medical images. After comparing several models, **EfficientNetB2** emerged as the best option due to its balance of speed, accuracy, and low memory usage. However, other models like ResNet50 and DenseNet121 also show strong performance and could be used depending on the specific requirements of the application.""")

    # 9. Reference Links
    st.header("References")
    st.write("""
    - [Deep Learning in Medical Imaging](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6631868/)
    - [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
    - [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
    - [DenseNet: Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
    - [VGG19: Very Deep Convolutional Networks](https://arxiv.org/abs/1409.1556)""")

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
