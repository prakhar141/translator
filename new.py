import os
import requests
import tempfile
import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

st.set_page_config(page_title="English to Hindi Translator", page_icon="🌍")
st.title("🌍 English to Hindi Translator 🇮🇳")

@st.cache_resource(show_spinner="🔄 Loading model & tokenizer...")
def load_model_and_tokenizer():
    # Model repo on Hugging Face
    repo_url = "https://huggingface.co/prakhar146/english-hindi-tf-model/resolve/main/"
    local_dir = os.path.join(tempfile.gettempdir(), "tf_model")
    os.makedirs(local_dir, exist_ok=True)

    files = [
        "config.json",
        "tf_model.h5",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "source.spm",
        "target.spm"
    ]

    # Download all files if not already
    for fname in files:
        fpath = os.path.join(local_dir, fname)
        if not os.path.exists(fpath):
            with open(fpath, "wb") as f:
                f.write(requests.get(repo_url + fname).content)

    # Load model and tokenizer
    model = TFAutoModelForSeq2SeqLM.from_pretrained(local_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_dir)

    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ✏️ User input
input_text = st.text_area("✏️ Enter English sentence", "Hello, how are you?")

# 🚀 Translate
if st.button("Translate"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        try:
            encoded = tokenizer([input_text], return_tensors="tf")
            output = model.generate(**encoded, max_length=128)
            translated = tokenizer.decode(output[0], skip_special_tokens=True)
            st.subheader("✅ Hindi Translation")
            st.success(translated)
        except Exception as e:
            st.error("❌ Translation failed. Check model/tokenizer compatibility.")
            st.exception(e)
