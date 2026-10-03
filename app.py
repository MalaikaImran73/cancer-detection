
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🩺",
    layout="centered"
)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "best_model.keras",
        compile=False
    )


# ------------------------------------------------------------
# LOAD CLASS NAMES
# ------------------------------------------------------------

@st.cache_data
def load_class_names():

    with open(
        "class_names.json",
        "r"
    ) as f:

        return json.load(f)


model = load_model()
class_names = load_class_names()


# ------------------------------------------------------------
# USER INTERFACE
# ------------------------------------------------------------

st.title("🩺 Skin Cancer Detection")

st.write(
    "Upload a skin lesion image to obtain a "
    "prediction from the trained deep learning model."
)

st.warning(
    "This application is for educational and research "
    "purposes only. It is NOT a medical diagnosis."
)


# ------------------------------------------------------------
# IMAGE UPLOAD
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize to model input size
    image_resized = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image_resized,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Model prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        predictions
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = predictions[
        predicted_index
    ]

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.subheader("Prediction")

    st.success(
        f"Predicted Class: {predicted_class}"
    )

    st.write(
        f"Confidence: {confidence:.2%}"
    )

    # --------------------------------------------------------
    # ALL CLASS PROBABILITIES
    # --------------------------------------------------------

    st.subheader("Class Probabilities")

    probability_data = {
        class_names[i]: float(predictions[i])
        for i in range(len(class_names))
    }

    st.bar_chart(
        probability_data
    )

    st.info(
        "Please consult a qualified healthcare professional "
        "for medical evaluation. This model should not be "
        "used as a substitute for professional diagnosis."
    )
