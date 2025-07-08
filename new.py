import streamlit as st
import tensorflow as tf
import tempfile
import requests
import os
from transformers import AutoTokenizer

# App layout
st.set_page_config(page_title="English to Hindi Translator", page_icon="🌍")
st.title("🌍 English to Hindi Translator 🇮🇳")

@st.cache_resource(show_spinner="🔄 Downloading and loading model...")
def load_model_and_tokenizer():
    # Load tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("prakhar146/english-hindi-tf-model")

    # Download tf_model.h5 from Hugging Face repo manually
    model_url = "https://huggingface.co/prakhar146/english-hindi-tf-model/resolve/main/tf_model.h5"
    tmp_model_path = os.path.join(tempfile.gettempdir(), "tf_model.h5")

    if not os.path.exists(tmp_model_path):
        with open(tmp_model_path, "wb") as f:
            f.write(requests.get(model_url).content)

    # Load the keras model
    model = tf.keras.models.load_model(tmp_model_path)

    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# User input
input_text = st.text_area("✏️ Enter English sentence", "Hello, is there anything for translation?")

# Translate on button click
if st.button("🚀 Translate"):
    if not input_text.strip():
        st.warning("Please enter something.")
    else:
        inputs = tokenizer([input_text.strip()], return_tensors="tf")
        outputs = model.generate(**inputs, max_length=128)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("✅ Hindi Translation")
        st.success(translation)
