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

    # ✅ Download model file
    model_path = os.path.join(tempfile.gettempdir(), "tf_model.h5")
    if not os.path.exists(model_path):
        response = requests.get(base_url + "tf_model.h5")
        with open(model_path, "wb") as f:
            f.write(response.content)

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error("❌ Failed to load the model. Make sure tf_model.h5 was saved with both structure + weights.")
        raise e

    # ✅ Download config files (Optional for future use)
    config_path = os.path.join(tempfile.gettempdir(), "config.json")
    gen_config_path = os.path.join(tempfile.gettempdir(), "generation_config.json")

    for name, path in [("config.json", config_path), ("generation_config.json", gen_config_path)]:
        if not os.path.exists(path):
            response = requests.get(base_url + name)
            with open(path, "wb") as f:
                f.write(response.content)

    # Load config files if needed
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        with open(gen_config_path, "r") as f:
            gen_config = json.load(f)
    except:
        config, gen_config = {}, {}

    return model, config, gen_config

model, config, generation_config = load_model_and_config()

# 📝 User input
input_text = st.text_area("✏️ Enter English sentence", "Hello, how are you?")

# 🔄 Translation (Assumes model is ready for string input)
if st.button("🚀 Translate"):
    if not input_text.strip():
        st.warning("Please enter something.")
    else:
        try:
            prediction = model.predict([input_text.strip()])
            # Show prediction (adjust if model gives token IDs etc.)
            st.subheader("✅ Hindi Translation")
            st.success(prediction[0])
        except Exception as e:
            st.error("❌ Something went wrong during prediction. Check model input format.")
            st.exception(e)
