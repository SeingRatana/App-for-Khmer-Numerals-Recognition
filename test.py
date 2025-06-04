# This is the full optimized version of your Streamlit Khmer Digit app
# All design elements are preserved; only performance-related improvements are applied inline

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import time
import random

# Page configuration
st.set_page_config(
    page_title="កម្មវិធីសម្គាល់លេខខ្មែរ",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache model load
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("digit_model.keras")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

# Khmer numeral translation
def to_khmer_number(num):
    khmer_digits = "០១២៣៤៥៦៧៨៩"
    return ''.join(khmer_digits[int(d)] if d.isdigit() else d for d in str(num))

# Cache image preprocessing
@st.cache_data
def preprocess_image(img):
    if img is None or img.size == 0:
        return None
    img = cv2.resize(img.astype(np.uint8), (28, 28))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

# Prediction wrapper
def predict(img):
    if model is None or img is None:
        return None, 0, None
    try:
        pred = model.predict(img)
        return np.argmax(pred), np.max(pred), pred
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, 0, None

# UI Header
st.markdown("""
    <style>
        .stApp { max-width: 1200px; margin: auto; }
        .game-stat { font-weight: bold; font-size: 1.2em; color: #6e8efb; }
        .canvas-container { display: flex; justify-content: center; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("កម្មវិធីសម្គាល់លេខខ្មែរ ✨")
st.markdown("សម្គាល់ ឬគូរលេខខ្មែរ!")

# Tabs
tab1, tab2 = st.tabs(["សម្គាល់លេខ", "លេងហ្គេម"])

# Recognition Tab
with tab1:
    input_method = st.radio("ជ្រើសរើសវិធីបញ្ចូល", ["បង្ហោះរូបភាព", "គូរលេខ"], horizontal=True)

    if input_method == "បង្ហោះរូបភាព":
        file = st.file_uploader("អាប់ឡូដរូបភាពលេខខ្មែរ", type=["png", "jpg", "jpeg"])
        if file:
            image = Image.open(file).convert("L")
            img_np = np.array(image)
            st.image(img_np, caption="រូបភាពដើម", width=150)
            img_proc = 255 - img_np
            st.image(img_proc, caption="រូបភាពបម្លែង", width=150)
            if st.button("🔍 សម្គាល់"):
                with st.spinner("កំពុងវិភាគ..."):
                    processed = preprocess_image(img_proc)
                    digit, conf, raw_pred = predict(processed)
                if raw_pred is not None:
                    st.success(f"លេខទស្សន៍ទាយ: {to_khmer_number(digit)} ({conf*100:.2f}%)")
                    st.bar_chart({to_khmer_number(i): float(p) for i, p in enumerate(raw_pred[0])})
    else:
        st.markdown("គូរលេខខ្មែរលើផ្ទាំងខាងក្រោម:")
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=20,
            stroke_color="#FFF",
            background_color="#000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas_rec"
        )
        if canvas_result.image_data is not None:
            img_data = canvas_result.image_data[:, :, 0].astype(np.uint8)
            if np.any(img_data):
                st.image(img_data, caption="មើលជាមុន", width=112)
                if st.button("🔍 សម្គាល់" , key="recog_btn"):
                    with st.spinner("កំពុងវិភាគ..."):
                        processed = preprocess_image(img_data)
                        digit, conf, raw_pred = predict(processed)
                    if raw_pred is not None:
                        st.success(f"លេខទស្សន៍ទាយ: {to_khmer_number(digit)} ({conf*100:.2f}%)")
                        st.bar_chart({to_khmer_number(i): float(p) for i, p in enumerate(raw_pred[0])})

# Game Tab
with tab2:
    if "game" not in st.session_state:
        st.session_state.game = {
            "score": 0,
            "equation": {},
            "active": False,
            "start_time": None
        }

    def generate_equation():
        a = random.randint(0, 9)
        b = random.randint(1, 9)
        op = random.choice(["+", "-", "*", "/"])
        result = a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else a // b
        return {"a": a, "b": b, "op": op, "res": result}

    game = st.session_state.game

    if not game["active"]:
        if st.button("🚀 ចាប់ផ្តើមលេងហ្គេម"):
            game.update({
                "active": True,
                "score": 0,
                "equation": generate_equation(),
                "start_time": time.time()
            })
            st.experimental_rerun()
    else:
        elapsed = int(time.time() - game["start_time"])
        if elapsed >= 120:
            st.warning(f"⌛ ហ្គេមចប់! ពិន្ទុសរុប: {game['score']}")
            if st.button("🔁 លេងម្តងទៀត"):
                game.update({"score": 0, "active": False, "start_time": None})
                st.experimental_rerun()
        else:
            eq = game["equation"]
            st.markdown(f"**{to_khmer_number(eq['a'])} {eq['op']} ? = {to_khmer_number(eq['res'])}**")
            canvas = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=20,
                stroke_color="#FFF",
                background_color="#000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas_game"
            )
            col1, col2 = st.columns(2)
            if col1.button("🔍 ពិនិត្យចម្លើយ"):
                img_data = canvas.image_data[:, :, 0].astype(np.uint8)
                if np.any(img_data):
                    processed = preprocess_image(img_data)
                    digit, _, _ = predict(processed)
                    if eq["op"] == "+":
                        correct = (eq["a"] + digit == eq["res"])
                    elif eq["op"] == "-":
                        correct = (eq["a"] - digit == eq["res"])
                    elif eq["op"] == "*":
                        correct = (eq["a"] * digit == eq["res"])
                    else:
                        correct = (digit != 0 and eq["a"] // digit == eq["res"])
                    if correct:
                        st.success("🎉 ត្រឹមត្រូវ! +1 ពិន្ទុ")
                        game["score"] += 1
                    else:
                        st.error("🤔 មិនត្រឹមត្រូវទេ")
                    game["equation"] = generate_equation()
                    st.experimental_rerun()
            if col2.button("⏭️ រំលង"):
                game["equation"] = generate_equation()
                st.experimental_rerun()
            st.markdown(f"**ពិន្ទុ៖** {game['score']} | **ម៉ោង៖** {max(0, 120 - elapsed)}s ⏳")
