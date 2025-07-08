import os
import requests
import tempfile
import streamlit as st
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

st.set_page_config(page_title="English to Hindi Translator", page_icon="🌍")
st.title("🌍 English to Hindi Translator 🇮🇳")

@st.cache_resource(show_spinner="🔄 Loading model and tokenizer...")
def load_model_and_tokenizer():
    base_url = "https://huggingface.co/prakhar146/english-hindi-tf-model/resolve/main/"
    model_dir = os.path.join(tempfile.gettempdir(), "tf_model")
    os.makedirs(model_dir, exist_ok=True)

    files_to_download = [
        "tf_model.h5",
        "config.json",
        "tokenizer_config.json"
    ]

    for file_name in files_to_download:
        file_path = os.path.join(model_dir, file_name)
        if not os.path.exists(file_path):
            url = base_url + file_name
            response = requests.get(url)
            with open(file_path, "wb") as f:
                f.write(response.content)

    # Load model and tokenizer directly from the folder
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ✏️ User input
input_text = st.text_area("Enter English sentence", "Hello, how are you?")

# 🚀 Translate
if st.button("Translate"):
    if not input_text.strip():
        st.warning("Please enter something.")
    else:
        try:
            inputs = tokenizer([input_text], return_tensors="tf")
            outputs = model.generate(**inputs, max_length=128)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

            st.subheader("✅ Hindi Translation")
            st.success(translation)
        except Exception as e:
            st.error("❌ Something went wrong during translation.")
            st.exception(e)
