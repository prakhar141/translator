import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

st.set_page_config(page_title="English to Hindi Translator", page_icon="🌍")
st.title("🌍 English to Hindi Translator 🇮🇳")

@st.cache_resource(show_spinner="🔄 Loading model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    model = TFAutoModelForSeq2SeqLM.from_pretrained("tf_model")
    return tokenizer, model

tokenizer, model = load_model()

input_text = st.text_area("✏️ Enter English sentence", "India is a beautiful country.")

if st.button("🚀 Translate"):
    if not input_text.strip():
        st.warning("Please enter something.")
    else:
        inputs = tokenizer([input_text.strip()], return_tensors="tf")
        outputs = model.generate(**inputs, max_length=128)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("✅ Hindi Translation")
        st.success(translation)
