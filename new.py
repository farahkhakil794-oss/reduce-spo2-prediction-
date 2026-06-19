import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# 1. إعدادات الصفحة
st.set_page_config(page_title="ICU SpO2 Monitor | AI Hospital", page_icon="🏥", layout="wide")

# 2. إعداد قائمة اختيار اللغة في القائمة الجانبية (Sidebar)
st.sidebar.markdown("### 🌐 Settings / الإعدادات")
lang = st.sidebar.radio("Select Language / اختر اللغة:", ["English", "العربية"])

# 3. قاموس الترجمة الديناميكي (Dynamic Translation Dictionary)
texts = {
    "English": {
        "header": "🏥 Intensive Care Unit - AI System (AI-ICU)",
        "subheader": "Real-time SpO2 Estimation System | Advanced Sensor Edition (11 Features)",
        "panel_title": "### 📋 Patient Vitals Input Panel",
        "sec1": "🫀 Blood Flow & Pulse",
        "sec2": "📊 Capacity & Stiffness",
        "sec3": "⚙️ Patient & Sensor Data",
        "btn": "🚀 Start Processing PPG Signals & Extract SpO2",
        "loading": "Processing vital signs with AI...",
        "res_title": "Vitals Stable | Real-time SpO2 Reading",
        "res_footer": "Calculated by high-precision XGBoost algorithm",
        "err_model": "⚠️ System Alert: Model file not found. Please check the server path.",
        "lbl_pwv": "Pulse Wave Velocity (PWV m/s)",
        "lbl_p2p": "Peak-to-Peak Interval (ms)",
        "lbl_pr": "Pulse Rate (BPM)",
        "lbl_dt": "Diastolic Time (ms)",
        "lbl_ai": "Augmentation Index (%)",
        "lbl_sa": "Systolic Peak Amplitude",
        "lbl_si": "Stiffness Index (m/s)",
        "lbl_pi": "Perfusion Index (%)",
        "lbl_age": "Patient Age",
        "lbl_noise": "Sensor Noise Level (dB)",
        "lbl_site": "Sensor Placement Site",
        "sites": ["Fingertip", "Wrist"]
    },
    "العربية": {
        "header": "🏥 وحدة العناية المركزة - نظام الذكاء الاصطناعي (AI-ICU)",
        "subheader": "نظام المراقبة اللحظية لتقدير نسبة الأكسجين (SpO2) | إصدار السنسور الفائق",
        "panel_title": "### 📋 لوحة إدخال الإشارات الحيوية للمريض",
        "sec1": "🫀 مؤشرات تدفق الدم والنبض",
        "sec2": "📊 مؤشرات السعة والصلابة",
        "sec3": "⚙️ بيانات المريض وحالة السنسور",
        "btn": "🚀 بدء تحليل إشارات الـ PPG واستخراج نسبة الأكسجين",
        "loading": "جاري معالجة الإشارات الحيوية بالذكاء الاصطناعي...",
        "res_title": "نبضات القلب مستقرة | القراءة اللحظية لـ SpO2",
        "res_footer": "تم الحساب بواسطة خوارزمية XGBoost بدقة فائقة",
        "err_model": "⚠️ تنبيه نظام: السيرفر لا يستطيع الوصول لملف الموديل الطبي.",
        "lbl_pwv": "سرعة موجة النبض (PWV m/s)",
        "lbl_p2p": "الفترة بين الذروتين (Peak-to-Peak ms)",
        "lbl_pr": "معدل النبض (Pulse Rate BPM)",
        "lbl_dt": "الزمن الانبساطي (Diastolic Time ms)",
        "lbl_ai": "مؤشر التضخيم (Augmentation Index %)",
        "lbl_sa": "سعة الذروة الانقباضية (Systolic Amp)",
        "lbl_si": "مؤشر الصلابة (Stiffness Index m/s)",
        "lbl_pi": "معامل الإرواء (Perfusion Index %)",
        "lbl_age": "عمر المريض (Age)",
        "lbl_noise": "مستوى التشويش (Noise dB)",
        "lbl_site": "مكان تركيب السنسور (Site)",
        "sites": ["Fingertip (إصبع)", "Wrist (معصم)"]
    }
}

# المتغير t يحتوي على نصوص اللغة المختارة حالياً
t = texts[lang]
text_dir = "ltr" if lang == "English" else "rtl"

# 4. تصميم CSS ديناميكي يتأقلم مع اللغة
st.markdown(f"""
    <style>
    .main-container {{ direction: {text_dir}; }}
    .main-header {{
        background-color: #0f172a; padding: 20px; border-radius: 10px;
        color: #f8fafc; text-align: center; font-family: sans-serif;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px; direction: {text_dir};
    }}
    .main-header h1 {{ margin: 0; font-size: 2.2rem; color: #38bdf8; }}
    .main-header p {{ margin: 5px 0 0 0; font-size: 1.1rem; color: #94a3b8; }}
    .result-box {{
        background-color: #ecfdf5; border-left: 8px solid #10b981; border-right: none;
        padding: 25px; border-radius: 10px; text-align: center;
        margin-top: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); direction: {text_dir};
    }}
    .result-value {{ font-size: 3.5rem; color: #047857; font-weight: bold; margin: 10px 0; }}
    .result-title {{ font-size: 1.5rem; color: #065f46; margin: 0; }}
    </style>
""", unsafe_allow_html=True)

# 5. الهيدر الرئيسي
st.markdown(f"""
    <div class="main-header">
        <h1>{t['header']}</h1>
        <p>{t['subheader']}</p>
    </div>
""", unsafe_allow_html=True)

# 6. تحميل الموديل بأمان
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'reduced_spo2_model.pkl')

if not os.path.exists(model_path):
    st.error(t['err_model'])
    st.stop()

@st.cache_resource
def load_medical_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

model = load_medical_model(model_path)

# 7. واجهة الإدخال
st.markdown(f"<div class='main-container'>", unsafe_allow_html=True)
st.write(t['panel_title'])
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.info(t['sec1'])
    pulse_wave_velocity = st.number_input(t['lbl_pwv'], value=8.64, format="%.2f")
    peak_to_peak = st.number_input(t['lbl_p2p'], value=855.90, format="%.2f")
    pulse_rate = st.number_input(t['lbl_pr'], value=70.0, format="%.1f")
    diastolic_time = st.number_input(t['lbl_dt'], value=521.90, format="%.2f")

with col2:
    st.warning(t['sec2'])
    augmentation_index = st.number_input(t['lbl_ai'], value=10.30, format="%.2f")
    systolic_amplitude = st.number_input(t['lbl_sa'], value=0.71, format="%.2f")
    stiffness_index = st.number_input(t['lbl_si'], value=8.67, format="%.2f")
    perfusion_index = st.number_input(t['lbl_pi'], value=4.26, format="%.2f")

with col3:
    st.success(t['sec3'])
    age = st.number_input(t['lbl_age'], min_value=1, max_value=120, value=50)
    noise_level = st.number_input(t['lbl_noise'], value=-37.60, format="%.2f")
    site_label = st.selectbox(t['lbl_site'], t['sites'])
    measurement_site = 0 if "Fingertip" in site_label else 1

st.divider()

# 8. زر تشغيل النظام
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    calculate_btn = st.button(t['btn'], use_container_width=True)

# 9. الترتيب الدقيق والحساب
if calculate_btn:
    with st.spinner(t['loading']):
        # الترتيب الآلي للمميزات لا يتأثر بتغيير اللغة
        features_ordered = np.array([[
            pulse_wave_velocity,      # 1
            peak_to_peak,             # 2
            augmentation_index,       # 3
            perfusion_index,          # 4
            pulse_rate,               # 5
            systolic_amplitude,       # 6
            noise_level,              # 7
            stiffness_index,          # 8
            age,                      # 9
            diastolic_time,           # 10
            measurement_site          # 11
        ]])
        
        prediction = model.predict(features_ordered)[0]
        
        # عرض النتيجة
        st.markdown(f"""
            <div class="result-box">
                <p class="result-title">{t['res_title']}</p>
                <p class="result-value">{prediction:.2f}%</p>
                <p style="color: #64748b; margin: 0;">{t['res_footer']}</p>
            </div>
        """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
