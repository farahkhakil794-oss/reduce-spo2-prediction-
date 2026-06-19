import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="SpO2 Medical Dashboard", layout="wide")

# العنوان بتنسيق بسيط ومضمون
st.title("المنظومة الذكية لتقدير نسبة الأكسجين (SpO2)")
st.subheader("نسخة السنسور المحسّنة - دقة فائقة بـ 11 ميزة")

# 2. تحميل الموديل المعتمد (تأكدي أن الملف في نفس فولدر الكود)
@st.cache_resource
def load_model():
    with open('reduced_spo2_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error("❌ لم يتم العثور على ملف 'reduced_spo2_model.pkl'. تأكدي من رفعه في نفس المجلد.")
    st.stop()

# 3. بناء الواجهة (11 صندوق إدخال فقط)
col1, col2 = st.columns(2)

with col1:
    pulse_wave_velocity = st.number_input("سرعة موجة النبض (m/s)", value=8.64)
    peak_to_peak = st.number_input("الفترة بين الذروتين (ms)", value=855.90)
    augmentation_index = st.number_input("مؤشر التضخيم (%)", value=10.30)
    perfusion_index = st.number_input("معامل الإرواء (%)", value=4.26)
    pulse_rate = st.number_input("معدل النبض (BPM)", value=70.0)
    systolic_amplitude = st.number_input("سعة الذروة الانقباضية", value=0.71)

with col2:
    noise_level = st.number_input("مستوى الضوضاء (dB)", value=-37.60)
    stiffness_index = st.number_input("مؤشر الصلابة (m/s)", value=8.67)
    age = st.number_input("العمر", value=50)
    diastolic_time = st.number_input("الزمن الانبساطي (ms)", value=521.90)
    site_label = st.selectbox("مكان القياس", ["Fingertip", "Wrist"])
    measurement_site = 0 if site_label == "Fingertip" else 1

# 4. زر التوقع
if st.button("🔮 احسب نسبة الأكسجين"):
    # ترتيب المدخلات بدقة الـ 11 ميزة
    features = np.array([[
        pulse_wave_velocity, peak_to_peak, augmentation_index,
        perfusion_index, pulse_rate, systolic_amplitude, 
        noise_level, stiffness_index, age, diastolic_time, measurement_site
    ]])
    
    prediction = model.predict(features)[0]
    
    # عرض النتيجة
    st.metric(label="نسبة الأكسجين المتوقعة (SpO2)", value=f"{prediction:.2f}%")
    st.success("تم حساب التوقع بنجاح!")
