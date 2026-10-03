import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CIFAR-10 CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cifar10_model.keras")
    return model


try:
    model = load_model()
except Exception as e:
    st.error("❌ Unable to load the model.")
    st.error(str(e))
    st.stop()


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):
    """
    Resize uploaded image to CIFAR-10 model input size
    and normalize pixel values.
    """

    image = image.convert("RGB")

    # CIFAR-10 images are 32x32
    image_resized = image.resize((32, 32))

    image_array = np.array(image_resized)

    # Normalize pixels from 0-255 to 0-1
    image_array = image_array.astype("float32") / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):
    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image, verbose=0)

    # Handle models that return logits instead of probabilities
    if np.max(predictions) > 1 or not np.isclose(
        np.sum(predictions[0]), 1.0, atol=1e-3
    ):
        predictions = tf.nn.softmax(predictions).numpy()

    predicted_index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(predictions[0][predicted_index]) * 100

    probabilities = predictions[0] * 100

    return predicted_class, confidence, probabilities


# ============================================================
# HEADER
# ============================================================

st.title("🧠 CIFAR-10 Image Classification")
st.markdown(
    """
    **Deep Learning Prediction Interface**

    Upload an image and the trained CNN model will classify it into
    one of the **10 CIFAR-10 categories**.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write("**Model:** CIFAR-10 Deep Learning Model")

    st.write("**Framework:** TensorFlow / Keras")

    st.write("**Input Size:** 32 × 32 pixels")

    st.write("**Number of Classes:** 10")

    st.divider()

    st.subheader("📋 Classes")

    for i, class_name in enumerate(CLASS_NAMES, start=1):
        st.write(f"{i}. {class_name}")


# ============================================================
# FILE UPLOADER
# ============================================================

st.subheader("📤 Upload an Image")

uploaded_file = st.file_uploader(
    "Choose a CIFAR-10 image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG, JPEG, or PNG image."
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # --------------------------------------------------------
    # IMAGE + INFORMATION COLUMNS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🖼️ Uploaded Image")

        st.image(
            image,
            caption="Original Uploaded Image",
            use_container_width=True
        )

    with col2:

        st.subheader("🔍 Image Information")

        st.write(
            f"**File Name:** `{uploaded_file.name}`"
        )

        st.write(
            f"**Image Size:** `{image.size[0]} × {image.size[1]}` pixels"
        )

        st.write(
            f"**Image Mode:** `{image.mode}`"
        )

        st.info(
            "The image will be resized to 32 × 32 pixels "
            "before prediction."
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button(
        "🚀 Predict Image",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing image..."):

            predicted_class, confidence, probabilities = predict_image(
                image
            )

        st.success("Prediction completed successfully!")

        # ----------------------------------------------------
        # PREDICTION RESULT
        # ----------------------------------------------------

        st.subheader("🎯 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                label="Predicted Class",
                value=predicted_class.upper()
            )

        with result_col2:

            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

        st.divider()

        # ----------------------------------------------------
        # CONFIDENCE MESSAGE
        # ----------------------------------------------------

        if confidence >= 80:

            st.success(
                f"✅ The model is highly confident that this image "
                f"is a **{predicted_class}**."
            )

        elif confidence >= 50:

            st.warning(
                f"⚠️ The model predicts **{predicted_class}**, "
                f"but confidence is moderate."
            )

        else:

            st.info(
                f"ℹ️ The model predicts **{predicted_class}**, "
                f"but the confidence is relatively low."
            )

        # ----------------------------------------------------
        # PROBABILITY TABLE
        # ----------------------------------------------------

        st.subheader("📊 Class Probability Distribution")

        probability_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability (%)": probabilities
        })

        probability_df = probability_df.sort_values(
            "Probability (%)",
            ascending=False
        )

        st.dataframe(
            probability_df.style.format({
                "Probability (%)": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # BAR CHART
        # ----------------------------------------------------

        st.subheader("📈 Prediction Confidence by Class")

        chart_df = probability_df.sort_values(
            "Probability (%)",
            ascending=True
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.barh(
            chart_df["Class"],
            chart_df["Probability (%)"]
        )

        ax.set_xlabel("Probability (%)")

        ax.set_ylabel("CIFAR-10 Class")

        ax.set_title("Model Prediction Probabilities")

        ax.set_xlim(0, 100)

        for index, value in enumerate(
            chart_df["Probability (%)"]
        ):
            ax.text(
                value + 1,
                index,
                f"{value:.2f}%"
            )

        plt.tight_layout()

        st.pyplot(fig)

        # ----------------------------------------------------
        # TOP 3 PREDICTIONS
        # ----------------------------------------------------

        st.subheader("🏆 Top 3 Predictions")

        top_3 = probability_df.head(3).reset_index(drop=True)

        for i, row in top_3.iterrows():

            rank = i + 1

            st.write(
                f"**#{rank} {row['Class'].upper()}** — "
                f"{row['Probability (%)']:.2f}%"
            )

            st.progress(
                min(
                    int(row["Probability (%)"]),
                    100
                )
            )

else:

    # --------------------------------------------------------
    # INITIAL INSTRUCTION
    # --------------------------------------------------------

    st.info(
        "👆 Upload an image above to start the prediction."
    )

    st.subheader("💡 How to Use")

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("### 1️⃣")
        st.write("Upload an image")

    with step2:
        st.markdown("### 2️⃣")
        st.write("Image preprocessing")

    with step3:
        st.markdown("### 3️⃣")
        st.write("Run deep learning model")

    with step4:
        st.markdown("### 4️⃣")
        st.write("View prediction")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Task 6 – Creating a Streamlit User Interface | "
    "CIFAR-10 Deep Learning Image Classifier"
)