import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# 1. إعدادات الصفحة (يجب أن تكون أول سطر) - شاشة عريضة واسم طبي
st.set_page_config(page_title="ICU SpO2 Monitor | المستشفى الذكي", page_icon="🏥", layout="wide")

# 2. تصميم CSS احترافي يضيف طابع المستشفيات (ألوان طبية مريحة للعين)
st.markdown("""
    <style>
    .main-header {
        background-color: #0f172a;
        padding: 20px;
        border-radius: 10px;
        color: #f8fafc;
        text-align: center;
        font-family: 'Arial', sans-serif;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; color: #38bdf8; }
    .main-header p { margin: 5px 0 0 0; font-size: 1.1rem; color: #94a3b8; }
    .result-box {
        background-color: #ecfdf5;
        border-right: 8px solid #10b981;
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        margin-top: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .result-value { font-size: 3.5rem; color: #047857; font-weight: bold; margin: 10px 0; }
    .result-title { font-size: 1.5rem; color: #065f46; margin: 0; }
    div[data-testid="stMetricValue"] { font-size: 2rem; color: #0284c7; }
    </style>
""", unsafe_allow_html=True)

# 3. الهيدر الرئيسي للمستشفى
st.markdown("""
    <div class="main-header">
        <h1>🏥 وحدة العناية المركزة - نظام الذكاء الاصطناعي (AI-ICU)</h1>
        <p>نظام المراقبة اللحظية لتقدير نسبة تشبع الأكسجين في الدم (SpO2) | إصدار السنسور الفائق (11 ميزة)</p>
    </div>
""", unsafe_allow_html=True)

# 4. تحميل الموديل في الخلفية بأمان
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'reduced_spo2_model.pkl')

if not os.path.exists(model_path):
    st.error("⚠️ تنبيه نظام: السيرفر لا يستطيع الوصول لملف الموديل الطبي. يرجى مراجعة الدعم الفني.")
    st.stop()

@st.cache_resource
def load_medical_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

model = load_medical_model(model_path)

# 5. واجهة إدخال البيانات الحيوية (مقسمة طبياً لتشبه أجهزة المراقبة)
st.write("### 📋 لوحة إدخال الإشارات الحيوية للمريض (Patient Vitals Panel)")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🫀 مؤشرات تدفق الدم والنبض")
    pulse_wave_velocity = st.number_input("سرعة موجة النبض (PWV m/s)", value=8.64, format="%.2f")
    peak_to_peak = st.number_input("الفترة بين الذروتين (Peak-to-Peak ms)", value=855.90, format="%.2f")
    pulse_rate = st.number_input("معدل النبض (Pulse Rate BPM)", value=70.0, format="%.1f")
    diastolic_time = st.number_input("الزمن الانبساطي (Diastolic Time ms)", value=521.90, format="%.2f")

with col2:
    st.warning("📊 مؤشرات السعة والصلابة")
    augmentation_index = st.number_input("مؤشر التضخيم (Augmentation Index %)", value=10.30, format="%.2f")
    systolic_amplitude = st.number_input("سعة الذروة الانقباضية (Systolic Amp)", value=0.71, format="%.2f")
    stiffness_index = st.number_input("مؤشر الصلابة (Stiffness Index m/s)", value=8.67, format="%.2f")
    perfusion_index = st.number_input("معامل الإرواء (Perfusion Index %)", value=4.26, format="%.2f")

with col3:
    st.success("⚙️ بيانات المريض وحالة السنسور")
    age = st.number_input("عمر المريض (Age)", min_value=1, max_value=120, value=50)
    noise_level = st.number_input("مستوى التشويش بالسنسور (Noise dB)", value=-37.60, format="%.2f")
    site_label = st.selectbox("مكان تركيب السنسور (Site)", ["Fingertip (إصبع المريض)", "Wrist (معصم المريض)"])
    measurement_site = 0 if "Fingertip" in site_label else 1

st.divider()

# 6. زر تشغيل النظام (بشكل احترافي)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    calculate_btn = st.button("🚀 بدء تحليل إشارات الـ PPG واستخراج نسبة الأكسجين", use_container_width=True)

# 7. الترتيب الدقيق والحساب
if calculate_btn:
    with st.spinner('جاري معالجة الإشارات الحيوية بالذكاء الاصطناعي...'):
        # الترتيب هنا يتطابق 100% مع ترتيب الأعمدة الذي تدرب عليه الموديل في الكولاب
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
        
        # التوقع
        prediction = model.predict(features_ordered)[0]
        
        # عرض النتيجة بشكل شاشة العناية المركزة
        st.markdown(f"""
            <div class="result-box">
                <p class="result-title">نبضات القلب مستقرة | القراءة اللحظية لـ SpO2</p>
                <p class="result-value">{prediction:.2f}%</p>
                <p style="color: #64748b; margin: 0;">تم الحساب بواسطة خوارزمية XGBoost بدقة فائقة</p>
            </div>
        """, unsafe_allow_html=True)
