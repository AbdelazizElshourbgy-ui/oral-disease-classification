import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image

st.set_page_config(page_title="Oral Disease Classifier", page_icon="🦷", layout="wide")

IMG = 224
class_names = json.load(open("class_names.json"))

# ألوان وأيقونة وشرح بسيط لكل مرض
INFO = {
    "Calculus":            ("#E9A23B", "🪨", "Hardened plaque (tartar) on the teeth."),
    "Gingivitis":          ("#E5566D", "🩸", "Red, swollen or bleeding gums."),
    "Mouth_Ulcer":         ("#8E6CEF", "🔴", "Painful sore inside the mouth."),
    "Tooth_Discoloration": ("#3DB5A6", "🎨", "Teeth that changed color."),
    "Hypodontia":          ("#4C8DF6", "🦷", "One or more teeth are missing from birth."),
    "Caries":              ("#F26B3A", "🕳️", "Tooth decay that creates cavities."),
}

# ---------- التصميم (CSS) ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Poppins', sans-serif; }

.stApp {
    background:
        radial-gradient(1000px 500px at 85% -10%, rgba(0,180,216,.28), transparent 60%),
        radial-gradient(800px 500px at -10% 20%, rgba(124,92,255,.25), transparent 60%),
        linear-gradient(180deg, #0A1124 0%, #0D1730 100%);
    color: #E8EEF9;
}
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2.2rem; max-width: 1150px;}

.hero {
    padding: 34px 38px; border-radius: 28px; margin-bottom: 26px;
    background: linear-gradient(120deg, rgba(0,180,216,.18), rgba(124,92,255,.18));
    border: 1px solid rgba(255,255,255,.12);
    backdrop-filter: blur(14px);
    box-shadow: 0 20px 60px rgba(0,0,0,.35);
}
.hero h1 {
    margin: 0; font-size: 2.6rem; font-weight: 800; line-height: 1.15;
    background: linear-gradient(90deg, #5CE1FF, #A78BFA 60%, #FF8FB1);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p {margin: 10px 0 0; color: #AAB8D4; font-size: 1.05rem;}

.glass {
    padding: 22px; border-radius: 24px;
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.12);
    backdrop-filter: blur(12px);
    box-shadow: 0 12px 40px rgba(0,0,0,.30);
}
.result-name {font-size: 2.1rem; font-weight: 800; margin: 4px 0 2px;}
.result-desc {color: #AAB8D4; margin-bottom: 14px;}
.conf-num {font-size: 3.2rem; font-weight: 800; line-height: 1;}
.conf-label {color: #8FA0C2; font-size: .9rem; margin-bottom: 6px;}

.bar-row {margin: 11px 0;}
.bar-top {display:flex; justify-content:space-between; font-size:.92rem; margin-bottom:5px;}
.bar-bg {height: 10px; background: rgba(255,255,255,.08); border-radius: 99px; overflow: hidden;}
.bar-fill {height: 100%; border-radius: 99px;}

.chip {
    display:inline-block; padding: 6px 14px; margin: 4px 6px 4px 0; border-radius: 99px;
    font-size: .85rem; background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.14); color:#DCE6FA;
}
.note {
    margin-top: 18px; padding: 12px 16px; border-radius: 14px; font-size: .85rem;
    background: rgba(255,193,7,.10); border: 1px solid rgba(255,193,7,.30); color:#FFE08A;
}

[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,.05);
    border: 2px dashed rgba(92,225,255,.45);
    border-radius: 22px; padding: 26px;
}
[data-testid="stFileUploader"] section:hover {border-color: #5CE1FF;}
[data-testid="stFileUploader"] button {
    background: linear-gradient(90deg, #00B4D8, #7C5CFF); color: #fff;
    border: none; border-radius: 12px; font-weight: 600;
}
[data-testid="stImage"] img {border-radius: 20px; box-shadow: 0 12px 40px rgba(0,0,0,.45);}
</style>
""", unsafe_allow_html=True)


# ---------- تحميل الموديل ----------
@st.cache_resource
def load_model():
    base = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(IMG, IMG, 3))
    inp = tf.keras.Input((IMG, IMG, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.load_weights("best.weights.h5")
    return model

with st.spinner("Loading model..."):
    model = load_model()


def pretty(name):
    return name.replace("_", " ")


# ---------- الواجهة ----------
st.markdown("""
<div class="hero">
  <h1>🦷 Oral Disease Classifier</h1>
  <p>Upload a photo of the mouth or teeth and the AI will tell you which condition it sees.</p>
</div>
""", unsafe_allow_html=True)

file = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if not file:
    chips = "".join(f'<span class="chip">{INFO[c][1]} {pretty(c)}</span>' for c in class_names)
    st.markdown(f'<div class="glass"><b>The model can detect:</b><br><br>{chips}</div>',
                unsafe_allow_html=True)
else:
    img = Image.open(file).convert("RGB")
    x = np.array(img.resize((IMG, IMG)), dtype="float32")[None]
    probs = model.predict(x, verbose=0)[0]
    top = int(np.argmax(probs))
    color, icon, desc = INFO.get(class_names[top], ("#5CE1FF", "🦷", ""))

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.image(img, use_container_width=True)

    with right:
        bars = ""
        for i in np.argsort(probs)[::-1]:
            c = INFO.get(class_names[i], ("#5CE1FF",))[0]
            bars += (
                f'<div class="bar-row">'
                f'<div class="bar-top"><span>{pretty(class_names[i])}</span><span>{probs[i]*100:.1f}%</span></div>'
                f'<div class="bar-bg"><div class="bar-fill" style="width:{probs[i]*100:.1f}%; background:{c};"></div></div>'
                f'</div>'
            )

        html = (
            f'<div class="glass">'
            f'<div style="font-size:2.4rem">{icon}</div>'
            f'<div class="result-name" style="color:{color}">{pretty(class_names[top])}</div>'
            f'<div class="result-desc">{desc}</div>'
            f'<div class="conf-label">Confidence</div>'
            f'<div class="conf-num" style="color:{color}">{probs[top]*100:.1f}%</div>'
            f'<hr style="border-color:rgba(255,255,255,.1); margin:18px 0 8px">'
            f'{bars}'
            f'<div class="note">This is a student project and not a medical diagnosis. See a dentist for real advice.</div>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
