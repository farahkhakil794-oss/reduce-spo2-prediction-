import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# إعدادات الصفحة
st.set_page_config(page_title="SpO2 Medical Dashboard", layout="wide")

st.title("المنظومة الذكية لتقدير نسبة الأكسجين (SpO2)")
st.info("نسخة السنسور المحسّنة - دقة فائقة بـ 11 ميزة")

# --- كود كشف المسارات (عشان نتأكد الموديل فين بالظبط) ---
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'reduced_spo2_model.pkl')

# التحقق من وجود الملف
if not os.path.exists(model_path):
    st.error(f"❌ لم يتم العثور على الموديل! المجلد الحالي يحتوي على: {os.listdir(current_dir)}")
    st.stop()

# تحميل الموديل
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

model = load_model(model_path)

# --- واجهة الإدخال (11 ميزة) ---
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

# --- زر التوقع ---
if st.button("🔮 احسب نسبة الأكسجين"):
    features = np.array([[
        pulse_wave_velocity, peak_to_peak, augmentation_index,
        perfusion_index, pulse_rate, systolic_amplitude, 
        noise_level, stiffness_index, age, diastolic_time, measurement_site
    ]])
    
    prediction = model.predict(features)[0]
    
    # عرض النتيجة
    st.success(f"النتيجة المتوقعة لنسبة الأكسجين (SpO2): {prediction:.2f}%")
