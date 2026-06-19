import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. إعدادات الصفحة والعنوان بتنسيق طبي جذاب
st.set_page_config(page_title="Interactive SpO2 Medical System", layout="wide")

st.markdown("""
    <div style="background-color: #1E3A8A; padding: 20px; border-radius: 10px; text-align: center; color: white;">
        <h1 style="margin: 0;">المنظومة الذكية المحدثة لتقدير نسبة الأكسجين (SpO2)</h1>
        <p style="font-size: 1.2rem; margin-top: 10px;">نسخة السنسور المحسّنة والمضغوطة - دقة فائقة بـ 11 ميزة حيوية فقط</p>
    </div>
""", unsafe_allowed_html=True)

# 2. تحميل الموديل الجديد (reduced_spo2_model.pkl)
# تأكدي أن الملف مرفوع في نفس الفولدر على GitHub بجانب هذا الملف
@st.cache_resource
def load_medical_model():
    with open('reduced_spo2_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_medical_model()
    st.success("✅ تم تحميل الموديل الذكي الجديد بنجاح (اصدار الـ 11 ميزة)!")
except Exception as e:
    st.error(f"❌ عذراً، لم نجد ملف الموديل الجديد 'reduced_spo2_model.pkl'. تأكدي من رفعه على GitHub. الخطأ: {e}")

# 3. بناء واجهة المدخلات (الـ 11 صندوق الأساسية فقط مرتبة في عمودين لراحة العين)
st.write("### 📋 العوامل الحيوية للمريض (XGBoost Inputs)")

col1, col2 = st.columns(2)

with col1:
    pulse_wave_velocity = st.number_input("سرعة موجة النبض (Pulse Wave Velocity m/s)", min_value=0.0, max_value=30.0, value=8.64)
    peak_to_peak = st.number_input("الفترة بين الذروتين (Peak-to-Peak Interval ms)", min_value=0.0, max_value=2000.0, value=855.90)
    augmentation_index = st.number_input("مؤشر التضخيم (Augmentation Index %)", min_value=-50.0, max_value=100.0, value=10.30)
    perfusion_index = st.number_input("معامل الإرواء (Perfusion Index %)", min_value=0.0, max_value=20.0, value=4.26)
    pulse_rate = st.number_input("معدل النبض (Pulse Rate BPM)", min_value=30.0, max_value=200.0, value=70.0)
    systolic_amplitude = st.number_input("سعة الذروة الانقباضية (Systolic Peak Amplitude)", min_value=0.0, max_value=5.0, value=0.71)

with col2:
    noise_level = st.number_input("مستوى الضوضاء (Noise Level dB)", min_value=-100.0, max_value=0.0, value=-37.60)
    stiffness_index = st.number_input("مؤشر الصلابة (Stiffness Index m/s)", min_value=0.0, max_value=30.0, value=8.67)
    age = st.number_input("العمر (Age)", min_value=1, max_value=120, value=50)
    diastolic_time = st.number_input("الزمن الانبساطي (Diastolic Time ms)", min_value=0.0, max_value=2000.0, value=521.90)
    
    measurement_site_label = st.selectbox("مكان القياس (Measurement Site)", ["Fingertip (إصبع)", "Wrist (معصم)"])
    # تحويل الاختيار النصي إلى رقمي (0 أو 1) ليتوافق مع ما تعلمه الموديل في الكولاب
    measurement_site = 0 if measurement_site_label == "Fingertip (إصبع)" else 1

# 4. زر التوقع وحساب النتيجة اللحظية
st.markdown("---")
if st.button("🔮 تشغيل محرك التوقع اللحظي لنسبة الأكسجين", use_container_width=True):
    
    # تجميع المدخلات في مصفوفة بنفس الترتيب الدقيق للأعمدة الـ 11
    input_features = [
        pulse_wave_velocity, peak_to_peak, augmentation_index,
        perfusion_index, pulse_rate, systolic_amplitude, 
        noise_level, stiffness_index, age, diastolic_time, measurement_site
    ]
    
    # عمل التوقع
    features_array = np.array([input_features])
    prediction = model.predict(features_array)[0]
    
    # عرض النتيجة بشكل طبي محترف جداً
    st.markdown(f"""
        <div style="background-color: #D1FAE5; border-left: 8px solid #10B981; padding: 20px; border-radius: 5px; text-align: center; margin-top: 20px;">
            <h2 style="color: #065F46; margin: 0;">النتيجة المتوقعة لنسبة الأكسجين في الدم (SpO2):</h2>
            <h1 style="color: #047857; font-size: 3.5rem; margin: 10px 0;">{prediction:.2f}%</h1>
            <p style="color: #065F46; font-weight: bold; margin: 0;">الحالة الطبية المستنتجة بناءً على التوقع والـ AI المعزز</p>
        </div>
    """, unsafe_allowed_html=True)
