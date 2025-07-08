import os
import requests
import tempfile
import streamlit as st
import tensorflow as tf
import json
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

st.set_page_config(page_title="English to Hindi Translator", page_icon="🌍")
st.title("🌍 English to Hindi Translator 🇮🇳")

@st.cache_resource(show_spinner="🔄 Loading model and tokenizer...")
def load_model_and_tokenizer():
    base_url = "https://huggingface.co/prakhar146/english-hindi-tf-model/resolve/main/"

    # ✅ Download model file
    model_path = os.path.join(tempfile.gettempdir(), "tf_model.h5")
    if not os.path.exists(model_path):
        response = requests.get(base_url + "tf_model.h5")
        with open(model_path, "wb") as f:
            f.write(response.content)

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error("❌ Failed to load model. It must be saved with both architecture and weights.")
        raise e

    # ✅ Load tokenizer (same as training)
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# 📝 User input
input_text = st.text_area("✏️ Enter English sentence", "Hello, how are you?")

# 🔄 Translate
if st.button("🚀 Translate"):
    if not input_text.strip():
        st.warning("Please enter something.")
    else:
        try:
            inputs = tokenizer([input_text], return_tensors="tf")
            output_ids = model.generate(**inputs, max_length=128)
            translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            st.subheader("✅ Hindi Translation")
            st.success(translation)
        except Exception as e:
            st.error("❌ Prediction failed. Possibly tokenizer or model mismatch.")
            st.exception(e)
