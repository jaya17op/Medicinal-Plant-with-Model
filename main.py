import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown

# Google Drive model file ID
file_id = '1ONXVIC1CPGKVKwCZXTX_7NC156ozy4aL'  # replace with your file ID
output = 'trained_model.h5'

# Download only if not already downloaded
if not os.path.exists(output):
    gdown.download(f"https://drive.google.com/file/d/1ONXVIC1CPGKVKwCZXTX_7NC156ozy4aL/view?usp=sharing", output, quiet=False)



#tensorflow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(224,224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar
st.sidebar.markdown("## 🌿 *Medicinal Plant Recognition*")
# st.sidebar.image("images/logo.png", use_column_width=True)  # Optional logo
st.sidebar.markdown("Welcome to the plant recognition dashboard! Select a page to get started.")

app_mode = st.sidebar.selectbox(
    "📌 Navigate to:",
    ["Home", "About Project", "Predictions"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")
# st.sidebar.markdown("[GitHub Repo](https://github.com/yourusername/project) | [Contact](mailto:youremail@example.com)")

#Main Page 
if app_mode == "Home":
    st.title("🌿 Recognition of Medicinal Plants")
    st.markdown("---")

    # Optional image/banner
    # st.image("images/medicinal_plant_banner.jpg", use_column_width=True, caption="Discover the power of nature with AI 🌱")

    st.markdown("""
    Welcome to the **Medicinal Plant Recognition System**!  
    This project uses advanced **Machine Learning** and **Deep Learning** techniques to identify various medicinal plants from images.

    #### 🔍 Features:
    - Upload plant images for instant prediction
    - Supports multiple medicinal plant species
    - Easy-to-use interface powered by Streamlit

    > "Let nature be your medicine and technology your guide."

    ---
    🔁 Use the sidebar to navigate between:
    - 📷 **Predictions**
    - 📚 **About**
    - 📈 **Model Insights**
    """)

    st.success("👈 Start by selecting a page from the sidebar!")


#about
elif app_mode == "About Project":
    st.title("📘 About the Medicinal Plant Recognition Project")
    st.markdown("---")

    st.header("🔬 Project Overview")
    st.markdown("""
    This project is designed to **automatically recognize medicinal plants** using advanced **Machine Learning (ML)** and **Deep Learning (DL)** techniques. It aims to preserve traditional medicine, boost pharmacological research, and increase accessibility to herbal healthcare through image-based plant identification.
    
    The system supports 8 commonly used medicinal plants:
    - 🌱 Aloe Vera  
    - 🌿 Curry Leaves  
    - 🌿 Indian Borage  
    - 🌿 Menthi  
    - 🌿 Patharchatta  
    - 🌿 Ranapala Plant  
    - 🌿 Rosemary  
    - 🌿 Tulasi
    """)

    st.header("📂 Dataset & Preprocessing")
    st.markdown("""
    **Datasets Used:**
    - 🖼️ *Custom Dataset* – 500+ real-world images per plant species

    **Preprocessing Techniques:**
    - Image resizing to 224x224 pixels
    - Pixel normalization (0–1)
    - Augmentation: Rotation (±25°), Flipping, Zooming
    - Label encoding using one-hot vectors
    """)

    st.header("🧠 Model Architecture & Techniques")
    st.markdown("""
    **Baseline Model:** CNN with 3 convolutional layers + Dense + Softmax  
    **Transfer Learning Models:**  
    - 🧩 VGG16  
    - ⚡ EfficientNetB0  
    - 📱 MobileNetV2  

    **Explainable AI (XAI):**
    - Grad-CAM visualization
    - Enhanced Attention-CNN for focused learning
    """)

    st.header("⚙️ Training & Evaluation")
    st.markdown("""
    - Optimizers: Adam, RMSprop  
    - Loss: Categorical Crossentropy  
    - Regularization: Dropout (0.5), L2 (0.001)  
    - Metrics: Accuracy, Precision, Recall, F1-Score  

    **Best Results:**  
    - EfficientNet: 95.4%  
    - Enhanced Attention-CNN: 95.1%  
    """)

    st.header("🚀 Deployment Stack")
    st.markdown("""
    - **Web**: Flask (backend), HTML/CSS/JS (frontend)  
    - **Mobile**: Flutter App (with camera & gallery support)  
    - **Database**: MySQL for storing predictions, logs, and plant info  
    - **Bonus Feature**: 🌿 *Recipe suggestions* for each plant
    """)

    st.header("🧪 Hardware & Optimization")
    st.markdown("""
    - Trained on NVIDIA RTX 3060 (12GB VRAM) with CUDA acceleration  
    - Reduced training time by ~70% using GPU  
    - Optimized for mobile using model quantization
    """)

    st.header("📌 Conclusion")
    st.markdown("""
    This end-to-end system proves the effectiveness of CNNs and transfer learning in recognizing medicinal plants. With explainability (XAI), web/mobile deployment, and practical features like recipe suggestions, this tool is both robust and user-friendly.

    **Future Enhancements:**
    - Real-time detection
    - Multilingual support
    - Expanding species recognition
    """)

    st.markdown("---")
    st.caption("📖 For more details, check out the full project report or contact the author.")


#Prediction Page
elif app_mode == "Predictions":
    st.title("🌿 Medicinal Plant Prediction")
    st.markdown("Upload an image of a **medicinal plant**, and the model will predict its class using advanced Machine Learning and Deep Learning techniques.")

    uploaded_image = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.info("Make sure the image is clear and focused on the plant.")

        if st.button("🔍 Predict"):
            st.write("🧠 Analyzing the image...")
            result_index = model_prediction(uploaded_image)

            # Reading labels
            with open("label.txt") as f:
                labels = [line.strip() for line in f.readlines()]

            predicted_label = labels[result_index]
            st.success(f"✅ The model predicts this is: **{predicted_label}**")

    else:
        st.warning("Please upload an image to proceed with prediction.")
