import streamlit as st
import tensorflow as tf
import tempfile
import requests
import os
from transformers import AutoTokenizer

st.set_page_config(page_title="English to Hindi Translator", page_icon="🌍")
st.title("🌍 English to Hindi Translator 🇮🇳")

@st.cache_resource(show_spinner="🔄 Loading model...")
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("prakhar146/english-hindi-tf-model")

    # Download .h5 model from HF
    model_url = "https://huggingface.co/prakhar146/english-hindi-tf-model/resolve/main/tf_model.h5"
    model_path = os.path.join(tempfile.gettempdir(), "tf_model.h5")

    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            f.write(requests.get(model_url).content)

    model = tf.keras.models.load_model(model_path)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

input_text = st.text_area("✏️ Enter English sentence", "Hello, is there anything for translation?")

if st.button("🚀 Translate"):
    if not input_text.strip():
        st.warning("Please enter something.")
    else:
        inputs = tokenizer([input_text.strip()], return_tensors="tf")
        outputs = model.generate(**inputs, max_length=128)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("✅ Hindi Translation")
        st.success(translation)
