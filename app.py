import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import time

# Page Configuration
st.set_page_config(
    page_title="MilkQual AI | Premium Quality Classifier",
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium CSS with Glassmorphism and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
    
    :root {
        --primary: #6366f1;
        --primary-glow: rgba(99, 102, 241, 0.5);
        --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        --card-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.1);
        --text-main: #f8fafc;
        --text-dim: #94a3b8;
    }

    /* Global Styles */
    .main {
        background: var(--bg-gradient);
        color: var(--text-main);
        font-family: 'Outfit', sans-serif;
    }
    
    [data-testid="stAppViewContainer"] {
        background: var(--bg-gradient);
    }

    /* Glassmorphism Containers */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 2rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 12px 48px 0 rgba(99, 102, 241, 0.15);
        transform: translateY(-4px);
    }

    /* Custom Header */
    .hero-section {
        text-align: center;
        padding: 4rem 1rem;
        animation: fadeInDown 1s ease-out;
    }
    
    .hero-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 4.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        letter-spacing: -2px;
    }
    
    .hero-subtitle {
        color: var(--text-dim);
        font-size: 1.25rem;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Input Controls Customization */
    div.stSlider > div[data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
    }
    
    .stSlider label, .stSelectbox label {
        color: var(--text-main) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
    }

    div[data-baseweb="select"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 3rem !important;
        border-radius: 16px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.6) !important;
    }

    /* Results section */
    .result-badge {
        padding: 0.5rem 1.5rem;
        border-radius: 100px;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: inline-block;
        margin-bottom: 1.5rem;
    }
    
    .status-high { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .status-medium { background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.3); }
    .status-low { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

    .stat-label { color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.25rem; }
    .stat-value { font-size: 2.5rem; font-weight: 800; color: #fff; }

    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Utility */
    .flex-center { display: flex; align-items: center; justify-content: center; flex-direction: column; }
    .section-header {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    /* Footer */
    .footer-container {
        text-align: center;
        padding: 4rem 1rem;
        border-top: 1px solid var(--glass-border);
        margin-top: 4rem;
        color: var(--text-dim);
    }
    
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_trained_model():
    try:
        return load('model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">MilkQual AI</div>
    <div class="hero-subtitle">
        Advanced Machine Learning for Real-Time Dairy Quality Classification. 
        Ensuring safety and excellence through XGBoost analysis.
    </div>
</div>
""", unsafe_allow_html=True)

main_col1, main_col2, main_col3 = st.columns([1, 8, 1])

with main_col2:
    # Input Area
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span>📊</span> Sample Parameters</div>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        ph = st.slider("pH Level", 3.0, 9.5, 6.6, 0.1, help="Acidity level of the milk sample")
        temp = st.slider("Temperature (°C)", 20, 90, 45, 1, help="Current temperature of the sample")
        colour = st.slider("Color Value", 240, 260, 255, 1)

    with col_b:
        taste = st.selectbox("Taste Profile", [0, 1], format_func=lambda x: "Optimal" if x == 1 else "Non-Optimal")
        odor = st.selectbox("Odor Profile", [0, 1], format_func=lambda x: "Optimal" if x == 1 else "Non-Optimal")
        fat = st.selectbox("Fat Content", [0, 1], format_func=lambda x: "High" if x == 1 else "Low")
        turbidity = st.selectbox("Turbidity", [0, 1], format_func=lambda x: "High" if x == 1 else "Low")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("RUN CLASSIFICATION"):
        if model:
            with st.spinner("Processing through XGBoost layers..."):
                time.sleep(1.2) # Aesthetic delay
                
                # Prepare data
                input_df = pd.DataFrame([[ph, temp, taste, odor, fat, turbidity, colour]], 
                                     columns=['ph', 'temperature', 'taste', 'odor', 'fat', 'turbidity', 'colour'])
                
                prediction = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0]
                confidence = max(proba) * 100
                
                # Result Mapping
                grades = {0: ("LOW", "status-low", "Critical quality issues detected. Not recommended for consumption."), 
                         1: ("MEDIUM", "status-medium", "Acceptable quality. Within standard domestic ranges."), 
                         2: ("HIGH", "status-high", "Premium quality detected. Meets all safety and nutrient standards.")}
                
                grade_name, grade_class, grade_desc = grades[prediction]
                
                st.markdown(f"""
                <div class="glass-card flex-center" style="margin-top: 3rem; background: rgba(99, 102, 241, 0.05);">
                    <div class="result-badge {grade_class}">{grade_name} QUALITY</div>
                    <div class="stat-label">CLASSIFICATION CONFIDENCE</div>
                    <div class="stat-value">{confidence:.1f}%</div>
                    <div style="color: var(--text-dim); text-align: center; margin-top: 1.5rem; max-width: 400px;">
                        {grade_desc}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Model not found. Please ensure model.pkl exists.")
            
    st.markdown('</div>', unsafe_allow_html=True)

    # Info Cards
    info1, info2 = st.columns(2)
    with info1:
        st.markdown("""
        <div class="glass-card" style="padding: 1.5rem;">
            <div style="font-weight: 700; margin-bottom: 0.5rem; color: #fff;">Model Architecture</div>
            <div style="font-size: 0.85rem; color: var(--text-dim);">
                Utilizing XGBoost (Extreme Gradient Boosting) with optimized hyper-parameters for 99%+ accuracy on validation sets.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with info2:
        st.markdown("""
        <div class="glass-card" style="padding: 1.5rem;">
            <div style="font-weight: 700; margin-bottom: 0.5rem; color: #fff;">Dataset Origin</div>
            <div style="font-size: 0.85rem; color: var(--text-dim);">
                Analyzed against 1000+ samples of milk quality parameters including biochemical and sensory attributes.
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer-container">
    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: #fff; margin-bottom: 1rem;">
        DairySecure Technologies © 2026
    </div>
    <div style="font-size: 0.8rem; letter-spacing: 1px;">
        POWERED BY XGBOOST & STREAMLIT
    </div>
</div>
""", unsafe_allow_html=True)
