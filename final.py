import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import time
import random
import os
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Set page config and theme
st.set_page_config(
    page_title="កម្មវិធីសម្គាល់លេខខ្មែរ", # Khmer Digit App
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Khmer Translations
translations = {
    # General UI
    "app_title": "កម្មវិធីសម្គាល់លេខខ្មែរ ✨", 
    "app_subtitle": "សម្គាល់ ឬគូរលេខខ្មែរ!", 
    "sidebar_drawing_controls": "🖌️ សម្រាប់គូរ",
    "sidebar_stroke_recognition": "កម្រាស់ (គូរដើម្បីសម្គាល់):",
    "sidebar_stroke_game": "កម្រាស់ក្នុងហ្គេម (គូរក្នុងហ្គេម):",
    "sidebar_model_loaded": "✅ បានផ្ទុកម៉ូដែល!",
    "sidebar_model_not_loaded": "⚠️ រកមិនឃើញម៉ូដែល!",
    "sidebar_choose_mode": "🕹️ ជ្រើសរើសរបៀប:",
    "sidebar_about_app_title": "ℹ️ អំពីកម្មវិធីនេះ",
    "sidebar_khmer_numerals_img_alt": "លេខខ្មែរ",
    "sidebar_khmer_numerals_caption": "លេខខ្មែរ (០-៩)",
    "sidebar_about_app_desc_1": "<b>កម្មវិធីសម្គាល់លេខខ្មែរ</b> 🇰🇭 ប្រើ CNN សម្រាប់លេខខ្មែរ (០-៩)។",
    "sidebar_modes_title": "របៀប:",
    "sidebar_mode_recognition": "<b>សម្គាល់៖</b> បញ្ចូលរូបភាព ឬគូរ។",
    "sidebar_mode_game": "<b>ហ្គេម៖</b> ដោះស្រាយលំហាត់ដោយគូរ (កំណត់ពេល ៦០វិនាទី/សំណួរ)។ ពិន្ទុ +/- ១។",

    # Recognition Mode
    "rec_mode_title": "🧐 ការសម្គាល់លេខខ្មែរ",
    "rec_input_method": "វិធីសាស្ត្របញ្ចូល:",
    "rec_upload_image_radio": "📁 បង្ហោះរូបភាព",
    "rec_draw_digit_radio": "✏️ គូរលេខ",
    "rec_upload_title": "📁 បង្ហោះរូបភាព",
    "rec_upload_uploader_label": "រូបភាពលេខខ្មែរ",
    "rec_upload_original_caption": "រូបភាពដើម",
    "rec_upload_processed_caption": "រូបភាពកែច្នៃ (បញ្ច្រាស)",
    "rec_upload_button": "🔍 សម្គាល់រូបភាពដែលបានបង្ហោះ",
    "rec_upload_placeholder": "🖼️ បង្ហោះរូបភាព។",
    "rec_draw_title": "✏️ គូរលេខខ្មែរ",
    "rec_draw_canvas_label": "ផ្ទាំងក្រណាត់ (គូរពណ៌សលើផ្ទៃខ្មៅ):",
    "rec_draw_clear_button": "🧹 សម្អាតផ្ទាំងក្រណាត់",
    "rec_draw_preview_title": "មើលជាមុន & សម្គាល់:",
    "rec_draw_preview_caption": "មើលជាមុនពីផ្ទាំងក្រណាត់ (R-Channel)",
    "rec_draw_recognize_button": "🔍 សម្គាល់លេខដែលបានគូរ",
    "rec_draw_info_empty": "☝️ សូមគូរលើផ្ទាំងក្រណាត់ បន្ទាប់មកប៊ូតុង 'សម្គាល់' នឹងបង្ហាញ។",
    "rec_draw_tips_title": "<b>គន្លឹះគូរ៖</b>",
    "rec_draw_tip_1": "គូរលេខមួយខ្ទង់ឲ្យច្បាស់។",
    "rec_draw_tip_2": "កែសម្រួលកម្រាស់เส้นនៅរបារចំហៀង។",

    # Game Mode
    "game_mode_title": "🧮 ការប្រកួតដោះស្រាយលំហាត់លេខខ្មែរ! 🔢",
    "game_start_button": "🚀 ចាប់ផ្តើមលេងហ្គេម!",
    "game_over_message": "⌛ ហ្គេមចប់! ពិន្ទុសរុប: {score} 🎉", 
    "game_play_again_button": "🔁 លេងម្តងទៀត?",
    "game_score_label": "ពិន្ទុ:",
    "game_time_label": "ម៉ោង:",
    "game_stop_button": "🛑 បញ្ឈប់",
    "game_stop_button_help": "បញ្ចប់ហ្គេមបច្ចុប្បន្ន",
    "game_equation_header": "គូរលេខខ្មែរដែលបាត់!",
    "game_check_answer_button": "🔍 ពិនិត្យចម្លើយ!",
    "game_skip_button": "⏭️ រំលង / សំណួរបន្ទាប់",
    "game_toast_correct": "🎉 ត្រឹមត្រូវ! +១ ពិន្ទុ។ (ទំនុកចិត្ត: {confidence:.1f}%)", 
    "game_toast_incorrect_base": "🤔 មិនត្រឹមត្រូវទេ! ពិន្ទុ -១។ ",
    "game_toast_incorrect_wrong_digit": "ចម្លើយត្រឹមត្រូវសម្រាប់ '?' គឺ <strong>{khmer_correct_b}</strong> ({correct_b})។", 
    "game_toast_incorrect_wrong_equation": "អ្នកបានគូរលេខត្រឹមត្រូវសម្រាប់ '?' ប៉ុន្តែវាមិនបានដោះស្រាយសមីការទេ។",
    "game_toast_incorrect_generic": "មានអ្វីមួយមិនត្រឹមត្រូវទេ។",
    "game_toast_no_digit_drawn": "✏️ សូមគូរលេខជាមុន!",
    "game_toast_canvas_data_unavailable": "✏️ ទិន្នន័យផ្ទាំងក្រណាត់មិនអាចប្រើបានទេ។ សូមព្យាយាមគូរ។",
    "game_toast_model_not_loaded": "🚨 រកមិនឃើញម៉ូដែលទេ។",
    "game_toast_cannot_recognize": "⚠️ មិនអាចសម្គាល់លេខបានទេ។",

    # Common
    "analyzing_spinner": "កំពុងវិភាគ...",
    "checking_spinner": "កំពុងពិនិត្យ...",
    "prediction_header": "📊 ប្រូបាប៊ីលីតេនៃការទស្សន៍ទាយ",
    "predicted_digit_label": "លេខដែលបានទស្សន៍ទាយ:",
    "confidence_label": "ទំនុកចិត្ត:",
}


# Custom CSS
def apply_custom_style():
    st.markdown("""
    <style>
        :root {
            --background-color: #f5f7f9; --card-background: #ffffff; --text-color: #333333;
            --secondary-text: #666666; --accent-color: #6e8efb;
            --accent-gradient: linear-gradient(135deg, #6e8efb, #a777e3);
            --border-color: #e6e6e6; --success-color: #4CAF50; --warning-color: #FFC107;
            --error-color: #F44336; --info-color: #2196F3;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --background-color: #0e1117; --card-background: #1a1c23; --text-color: #f1f1f1;
                --secondary-text: #b0b0b0; --border-color: #2d3748;
            }
        }
        .main { background-color: var(--background-color); color: var(--text-color); }
        .card { background-color: var(--card-background); border-radius: 10px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem; color: var(--text-color); border: 1px solid var(--border-color); }
        h1,h2,h3,h4,p,li,a { color: var(--text-color); }
        .secondary-text { color: var(--secondary-text); }
        .stApp { max-width: 1200px; margin: 0 auto; }
        .app-header { font-family: 'Arial', sans-serif; background: var(--accent-gradient); padding: 1.5rem; border-radius: 10px; color: white !important; text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .app-header h1, .app-header p { color: white !important; }
        .prediction-result { font-size: 1.1em; font-weight: bold; text-align: center; padding: 15px; border-radius: 10px; margin-top: 20px; color: var(--text-color); }
        /* .game-feedback classes are no longer used for primary game feedback, st.toast is */
        .game-stat { font-size: 1.2em; font-weight: bold; color: var(--accent-color); margin-right: 20px; }
        .footer { text-align: center; margin-top: 2rem; padding: 1rem; font-size: 0.8rem; color: var(--secondary-text); }
        .method-selector { margin: 1rem auto 2rem auto; max-width: 600px; text-align: center; }
        .stRadio > div { display: flex; justify-content: center; }
        .stRadio > div > label { background-color: rgba(110,142,251,0.1); border-radius: 10px; padding: 10px 20px; margin: 0 10px; font-weight: bold; transition: all 0.3s; min-width: 180px; text-align: center; border: 1px solid var(--border-color); }
        .stButton > button { background-color: var(--accent-color); color: white; font-weight: bold; border-radius: 5px; padding: 0.5rem 1rem; border: none; transition: background-color 0.3s ease; }
        .stButton > button:hover { opacity: 0.9; }
        .stButton > button[kind="primary"] { background: var(--accent-gradient); font-size: 1.1em; }
        .stButton > button.secondary { background-color: var(--secondary-text); color: white; }
        .stButton > button.stop-game-btn { background-color: var(--error-color); color: white; } 
        .stButton > button.secondary:hover, .stButton > button.stop-game-btn:hover { opacity: 0.8; }
        .info-box { background-color: rgba(33,150,243,0.1); border-left: 6px solid var(--info-color); padding: 10px; margin-top: 20px; border-radius: 0 5px 5px 0; }
        .placeholder { background-color: rgba(0,0,0,0.05); border: 2px dashed var(--border-color); border-radius: 10px; padding: 40px; text-align: center; margin: 20px 0; }
        .canvas-container { 
            border: none; 
            padding: 0px; 
            margin-bottom: 15px; 
            background-color: transparent !important; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
        }
        .css-1s3fmew, .css-1x8cf1d { background-color: rgba(110,142,251,0.1); border: 1px dashed var(--border-color); padding: 10px; border-radius: 10px; }
        .sidebar .sidebar-content, [data-testid="stSidebar"] { background-color: var(--card-background); border-right: 1px solid var(--border-color); }
        div[data-testid="stBlock"] div.stBlock div[class$="ChartContainer"] { background-color: transparent !important; }
    </style>
    """, unsafe_allow_html=True)
apply_custom_style()

# --- Sidebar Controls ---
st.sidebar.markdown("---")
st.sidebar.subheader(translations["sidebar_drawing_controls"])
stroke_width_recognition = st.sidebar.slider(translations["sidebar_stroke_recognition"], 5, 50, 20, 1)
stroke_width_game = st.sidebar.slider(translations["sidebar_stroke_game"], 5, 50, 25, 1)

# --- Helper Functions ---
def to_khmer_number(number_str):
    number_str = str(number_str)
    khmer_digits = "០១២៣៤៥៦៧៨៩"
    return ''.join(khmer_digits[int(d)] if d.isdigit() else d for d in number_str)

@st.cache_resource
def load_model_cached():
    try: model = tf.keras.models.load_model('digit_model.keras'); return model
    except Exception as e: st.error(f"🚨 Could not load 'digit_model.keras': {e}."); return None
model = load_model_cached()
model_loaded = model is not None

if model_loaded: st.sidebar.success(translations["sidebar_model_loaded"])
else: st.sidebar.warning(translations["sidebar_model_not_loaded"])

def preprocess_image_for_model(img_array_2d_grayscale_uint8, debug_source_name="image"):
    if img_array_2d_grayscale_uint8 is None or img_array_2d_grayscale_uint8.size == 0: return None
    img_uint8 = img_array_2d_grayscale_uint8.astype(np.uint8)
    img_resized = cv2.resize(img_uint8, (28, 28), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype('float32') / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=(0, -1))
    return img_expanded

def predict_digit_from_processed_img(processed_img_expanded):
    if not model_loaded or processed_img_expanded is None: return None,0,None
    try:
        raw_prediction = model.predict(processed_img_expanded)
        return np.argmax(raw_prediction), np.max(raw_prediction), raw_prediction
    except Exception as e: st.error(f"Predict Error: {e}"); return None,0,None

def display_prediction_results(full_prediction_output): # For recognition mode
    if full_prediction_output is None: st.warning("No prediction output."); return
    pred_cls, conf = np.argmax(full_prediction_output), np.max(full_prediction_output) * 100
    khmer_pred = to_khmer_number(pred_cls)
    bg_style = f"background-color: var(--{'success' if conf > 80 else 'warning' if conf > 50 else 'error'}-color)33; border-left: 6px solid var(--{'success' if conf > 80 else 'warning' if conf > 50 else 'error'}-color);"
    st.markdown(f"""<div class="prediction-result" style="{bg_style}">
        <div>{translations["predicted_digit_label"]} <span style="font-size:42px;">{khmer_pred}</span> ({pred_cls})</div>
        <div>{translations["confidence_label"]} {conf:.2f}%</div></div>""", unsafe_allow_html=True)
    st.subheader(translations["prediction_header"])
    st.bar_chart({to_khmer_number(i):float(p) for i,p in enumerate(full_prediction_output[0])})

def generate_equation():
    a = random.randint(0, 9)
    b = random.randint(0, 9)
    op = random.choice(["+", "-", "*", "/"])

    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    else:
        b = random.randint(1, 9)
        result = random.randint(0, 9)
        a = b * result

    return {"a": a, "b": b, "op": op, "result": result}

# --- App Session State Initialization ---
if 'recognition_canvas_key' not in st.session_state: st.session_state.recognition_canvas_key = "canvas_rec_0"
if 'game_canvas_key' not in st.session_state: st.session_state.game_canvas_key = "canvas_game_init"
# Game state
if 'game_active' not in st.session_state: st.session_state.game_active = False
if 'game_over' not in st.session_state: st.session_state.game_over = False
if 'game_score' not in st.session_state: st.session_state.game_score = 0
if 'game_question_start_time' not in st.session_state: st.session_state.game_question_start_time = time.time()
if 'game_start_time' not in st.session_state: st.session_state.game_start_time = None  # <-- ADD THIS LINE
if 'game_time_limit_per_question' not in st.session_state: st.session_state.game_time_limit_per_question = 60
if "equation" not in st.session_state: st.session_state.equation = generate_equation()

def initialize_new_game_session():
    st.session_state.equation = generate_equation()
    st.session_state.game_question_start_time = time.time()
    st.session_state.game_start_time = time.time()  # ✅ Set game timer start here
    st.session_state.game_score = 0
    st.session_state.game_over = False
    st.session_state.game_active = True
    st.session_state.game_canvas_key = "canvas_game_" + str(time.time())
    st.rerun()


def start_new_game_question(increment_score=False, decrement_score=False):
    if increment_score:
        st.session_state.game_score += 1
    
    st.session_state.equation = generate_equation()
    st.session_state.game_question_start_time = time.time()
    st.session_state.game_canvas_key = "canvas_game_" + str(time.time())
    st.rerun()

def end_game():
    st.session_state.game_over = True
    st.session_state.game_active = False
    if "last_refresh_time" in st.session_state:
        del st.session_state.last_refresh_time
    st.rerun()

# --- App Header & Main Logic ---
st.markdown(f"<div class='app-header'><h1>{translations['app_title']}</h1><p>{translations['app_subtitle']}</p></div>", unsafe_allow_html=True)
app_mode = st.sidebar.radio(translations["sidebar_choose_mode"], [translations["rec_mode_title"], translations["game_mode_title"].split("!")[0]]) # Use shorter names for radio

# ========================= DIGIT RECOGNITION MODE =========================
if app_mode == translations["rec_mode_title"]:
    st.session_state.game_active = False 
    st.markdown(f"<div class='card'><h2>{translations['rec_mode_title']}</h2></div>", unsafe_allow_html=True)
    st.markdown('<div class="method-selector">', unsafe_allow_html=True)
    input_method = st.radio(translations["rec_input_method"], [translations["rec_upload_image_radio"], translations["rec_draw_digit_radio"]], horizontal=True, key="rec_input_method_key")
    st.markdown('</div>', unsafe_allow_html=True)

    if input_method == translations["rec_upload_image_radio"]:
        st.markdown(f"<div class='card'><h3>{translations['rec_upload_title']}</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(translations["rec_upload_uploader_label"], type=["jpg","jpeg","png"], key="img_uploader_key")
        if uploaded_file:
            col1_up, col2_up = st.columns([1,1])
            with col1_up:
                pil_img = Image.open(uploaded_file).convert("L")
                st.image(pil_img, caption=translations["rec_upload_original_caption"], use_column_width=True)
            img_np_orig = np.array(pil_img)
            img_np_inv_model = 255 - img_np_orig 
            with col2_up:
                st.image(img_np_inv_model, caption=translations["rec_upload_processed_caption"], use_column_width=True)
            if st.button(translations["rec_upload_button"], type="primary", key="pred_upload_btn_key", use_container_width=True):
                if model_loaded:
                    with st.spinner(translations["analyzing_spinner"]):
                        processed = preprocess_image_for_model(img_np_inv_model, "UPLOAD")
                        _, _, raw_pred = predict_digit_from_processed_img(processed)
                    display_prediction_results(raw_pred)
                else: st.error(translations["sidebar_model_not_loaded"])
        else: st.markdown(f"<div class='placeholder'><p style='font-size:1.2em;'>{translations['rec_upload_placeholder']}</p></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else: # Draw Digit
        st.markdown(f"<div class='card'><h3>{translations['rec_draw_title']}</h3>", unsafe_allow_html=True)
        col1_cv, col2_act = st.columns([1,1]) 
        with col1_cv:
            st.markdown(f"<p style='text-align:center;font-weight:bold;'>{translations['rec_draw_canvas_label']}</p>", unsafe_allow_html=True)
            st.markdown('<div class="canvas-container" style="margin:0 auto 15px auto;">', unsafe_allow_html=True) 
            cv_rec_data = st_canvas(fill_color="rgba(0,0,0,0)",stroke_width=stroke_width_recognition,stroke_color="#FFF",background_color="#000",height=280,width=280,drawing_mode="freedraw",key=st.session_state.recognition_canvas_key)
            st.markdown('</div>', unsafe_allow_html=True)
            def clear_rec_cv_action():
                parts=st.session_state.recognition_canvas_key.split('_');base,num="_".join(parts[:-1]),int(parts[-1])
                st.session_state.recognition_canvas_key=f"{base}_{num+1}"; st.rerun()
            st.button(translations["rec_draw_clear_button"], key="clear_rec_btn_key", on_click=clear_rec_cv_action, use_container_width=True)
        with col2_act:
            act_placeholder = st.container()
            is_drawing = False
            if cv_rec_data.image_data is not None and np.any(cv_rec_data.image_data[:,:,0]>0): is_drawing=True
            if is_drawing:
                with act_placeholder:
                    st.markdown(f"<h5>{translations['rec_draw_preview_title']}</h5>")
                    img_arr_cv = cv_rec_data.image_data[:,:,0].astype(np.uint8)
                    st.image(cv2.resize(img_arr_cv,(112,112)),translations["rec_draw_preview_caption"],112)
                    if st.button(translations["rec_draw_recognize_button"], type="primary", key="rec_draw_btn_key", use_container_width=True):
                        if model_loaded:
                            with st.spinner(translations["analyzing_spinner"]):
                                processed = preprocess_image_for_model(img_arr_cv,"CANVAS_REC")
                                _,_,raw_pred = predict_digit_from_processed_img(processed)
                            display_prediction_results(raw_pred)
                        else: st.error(translations["sidebar_model_not_loaded"])
            else:
                with act_placeholder: st.info(translations["rec_draw_info_empty"])
            st.markdown(f"""<div class="info-box" style="margin-top:20px;"><p>{translations['rec_draw_tips_title']}</p><ul>
                <li>{translations['rec_draw_tip_1']}</li><li>{translations['rec_draw_tip_2']}</li></ul></div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ========================= ARITHMETIC GAME MODE =========================
elif app_mode == translations["game_mode_title"].split("!")[0]:
    st.markdown(f"<div class='card'><h2>{translations['game_mode_title']}</h2></div>", unsafe_allow_html=True)
    # 🔁 Manual timer refresh every second during active game
    if st.session_state.game_active and not st.session_state.game_over:
        if "last_refresh_time" not in st.session_state:
            st.session_state.last_refresh_time = time.time()
        if time.time() - st.session_state.last_refresh_time >= 1:
            st.session_state.last_refresh_time = time.time()
            st.rerun()

    if not st.session_state.game_active and not st.session_state.game_over:
        if st.button(translations["game_start_button"], use_container_width=True, type="primary", key="start_game_btn"):
            initialize_new_game_session()
    elif st.session_state.game_over:
        st.error(translations["game_over_message"].format(score=st.session_state.game_score))
        if st.button(translations["game_play_again_button"], use_container_width=True, type="primary", key="play_again_btn"):
            initialize_new_game_session()
    else: 
        eq = st.session_state.equation
        khmer_a, khmer_result = to_khmer_number(eq['a']), to_khmer_number(eq['result'])

        time_now = time.time()
        total_game_time = 60  # Total allowed game time
        time_elapsed_total = time_now - st.session_state.game_start_time
        time_remaining_total = total_game_time - time_elapsed_total

        if time_remaining_total <= 0:
            end_game()

        col_score_disp, col_timer_disp, col_stop_game = st.columns([2,2,1])
        with col_score_disp:
            st.markdown(f"<span class='game-stat'>{translations['game_score_label']} {st.session_state.game_score}</span>", unsafe_allow_html=True)
        with col_timer_disp:
            timer_display_placeholder = st.empty() 
            timer_display_placeholder.markdown(f"<span class='game-stat' style='text-align:right;'>{translations['game_time_label']} {max(0, int(time_remaining_total))}s ⏳</span>", unsafe_allow_html=True)
        with col_stop_game:
            if st.button(translations["game_stop_button"], key="stop_game_btn", help=translations["game_stop_button_help"], use_container_width=True, type="secondary"):
                end_game()
        
        st.markdown("---")
        st.markdown(f"""<div class="card" style="text-align:center;margin-bottom:1.5rem;background:var(--accent-gradient);color:white;"><h3 style="color:white;font-size:2em;margin:0;">
            {khmer_a} &nbsp; {eq['op']} &nbsp; <span style="border:3px dashed white;padding:0px 15px;border-radius:8px;background:rgba(0,0,0,0.2);">{to_khmer_number("?")}</span> &nbsp; = &nbsp; {khmer_result}
            </h3><p style="color:white;margin-top:5px;">{translations['game_equation_header']}</p></div>""", unsafe_allow_html=True)
        
        col1_game_cv, col2_game_btns = st.columns([1,1])
        with col1_game_cv:
            st.markdown('<div class="canvas-container" style="margin:0 auto 15px auto;">', unsafe_allow_html=True)
            cv_game_data = st_canvas(fill_color="rgba(0,0,0,0)",stroke_width=stroke_width_game,stroke_color="#FFF",background_color="#000",height=280,width=280,drawing_mode="freedraw",key=st.session_state.game_canvas_key)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2_game_btns:
            st.write(" ") 
            check_ans_btn_clicked = st.button(translations["game_check_answer_button"], type="primary", key="chk_game_ans_btn", use_container_width=True)
            new_q_btn_clicked = st.button(translations["game_skip_button"], key="new_q_game_btn", use_container_width=True)

        if new_q_btn_clicked: 
            start_new_game_question() 

        if check_ans_btn_clicked:
            if cv_game_data.image_data is not None:
                img_arr_game_cv = cv_game_data.image_data[:,:,0].astype(np.uint8)
                if np.any(img_arr_game_cv > 0): 
                    if model_loaded:
                        with st.spinner(translations["checking_spinner"]):
                            processed = preprocess_image_for_model(img_arr_game_cv, "CANVAS_GAME")
                            pred_digit, conf, _ = predict_digit_from_processed_img(processed)
                        
                        if pred_digit is not None: 
                            user_in, actual_a, op, expected_rhs, correct_b = pred_digit, eq["a"], eq["op"], eq["result"], eq["b"]

                            # Default to False
                            user_solves_equation = False

                            # Determine if the equation is correct
                            if op == "+":
                                user_solves_equation = (actual_a + user_in == expected_rhs)
                            elif op == "-":
                                user_solves_equation = (actual_a - user_in == expected_rhs)
                            elif op == "*":
                                user_solves_equation = (actual_a * user_in == expected_rhs)
                            elif op == "/":
                                if user_in == 0:
                                    user_solves_equation = False
                                elif actual_a == 0 and expected_rhs == 0:
                                    # 0 / x = 0 is valid for any x != 0
                                    user_solves_equation = True
                                elif expected_rhs * user_in == actual_a and actual_a % user_in == 0:
                                    user_solves_equation = True
                                else:
                                    user_solves_equation = False

                            # Adjust what we consider a "correct digit"
                            if actual_a == 0 and op == "/" and expected_rhs == 0:
                                is_digit_correct = user_in in range(1, 10)  # Accept any digit 1–9
                            elif actual_a == 0 and op == "*" and expected_rhs == 0:
                                is_digit_correct = user_in in range(0, 10)  # Accept any digit 0–9
                            else:
                                is_digit_correct = user_in == correct_b


                            # Handle result
                            khmer_user = to_khmer_number(user_in)
                            khmer_correct_b = to_khmer_number(correct_b)

                            if is_digit_correct and user_solves_equation:
                                st.toast(translations["game_toast_correct"].format(confidence=conf * 100), icon="🎉")
                                st.balloons()
                                time.sleep(0.5)
                                start_new_game_question(increment_score=True)
                            else:
                                fbk_msg_key = "game_toast_incorrect_base"
                                if not is_digit_correct:
                                    fbk_details = translations["game_toast_incorrect_wrong_digit"].format(
                                        khmer_correct_b=khmer_correct_b, correct_b=correct_b
                                    )
                                elif not user_solves_equation:
                                    fbk_details = translations["game_toast_incorrect_wrong_equation"]
                                else:
                                    fbk_details = translations["game_toast_incorrect_generic"]
                                st.toast(translations[fbk_msg_key] + fbk_details, icon="🤔")
                                time.sleep(0.5)
                                start_new_game_question(decrement_score=True)

                        else: st.toast(translations["game_toast_cannot_recognize"], icon="⚠️")
                    else: st.toast(translations["game_toast_model_not_loaded"], icon="🚨")
                else: 
                    st.toast(translations["game_toast_no_digit_drawn"], icon="✏️")
            else: 
                st.toast(translations["game_toast_canvas_data_unavailable"], icon="✏️")
            # No st.rerun() here explicitly, start_new_game_question will handle it.

# --- Sidebar Info & Footer ---
with st.sidebar:
    st.markdown("---")
    st.markdown(f"<h3 style='text-align:center;'>{translations['sidebar_about_app_title']}</h3>", unsafe_allow_html=True)
    st.markdown(f"""<div style="text-align:center;padding:10px;"><img src="https://upload.wikimedia.org/wikipedia/commons/4/48/Khmer_numerals.svg" width="230" alt="{translations['sidebar_khmer_numerals_img_alt']}"><p style="font-size:0.8rem;margin-top:5px;" class="secondary-text">{translations['sidebar_khmer_numerals_caption']}</p></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"""<div style="padding:0 10px;"><p>{translations['sidebar_about_app_desc_1']}</p>
    <h4>{translations['sidebar_modes_title']}</h4><ul><li>{translations['sidebar_mode_recognition']}</li><li>{translations['sidebar_mode_game']}</li></ul>
    </div>""", unsafe_allow_html=True)