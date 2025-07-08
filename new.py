import os
import requests
import tempfile
import streamlit as st
import tensorflow as tf
import json

st.set_page_config(page_title="English to Hindi Translator", page_icon="🌍")
st.title("🌍 English to Hindi Translator 🇮🇳")

@st.cache_resource(show_spinner="🔄 Loading model and config...")
def load_model_and_config():
    base_url = "https://huggingface.co/prakhar146/english-hindi-tf-model/resolve/main/"

    # Download model
    model_path = os.path.join(tempfile.gettempdir(), "tf_model.h5")
    if not os.path.exists(model_path):
        r = requests.get(base_url + "tf_model.h5")
        with open(model_path, "wb") as f:
            f.write(r.content)

    model = tf.keras.models.load_model(model_path)

    # Download config files (if you need them for anything, optional)
    config_path = os.path.join(tempfile.gettempdir(), "config.json")
    gen_config_path = os.path.join(tempfile.gettempdir(), "generation_config.json")

    for url, path in [(base_url + "config.json", config_path), (base_url + "generation_config.json", gen_config_path)]:
        if not os.path.exists(path):
            r = requests.get(url)
            with open(path, "wb") as f:
                f.write(r.content)

    # If needed later
    with open(config_path) as f:
        config = json.load(f)

    with open(gen_config_path) as f:
        generation_config = json.load(f)

    return model, config, generation_config

model, config, generation_config = load_model_and_config()

# UI for user input
input_text = st.text_area("✏️ Enter English sentence", "Hello, how are you?")

if st.button("🚀 Translate"):
    if not input_text.strip():
        st.warning("Please enter something.")
    else:
        # Assume model expects input as plain string batch
        prediction = model.predict([input_text.strip()])
        
        # If your model outputs token IDs or characters, you will need to decode them manually
        st.subheader("✅ Hindi Translation")
        st.success(prediction[0])  # Adjust this if model output is more complex
