import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(
    page_title="Milk Quality Classifier | AI-Powered Analysis",
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS Design - Clean Centered Layout
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Center Container */
    .block-container {
        max-width: 900px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }
    
    /* Header Card */
    .header-card {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .header-card h1 {
        color: #2d3748;
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .header-card p {
        color: #718096;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Input Card */
    .input-card {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }
    
    .section-title {
        color: #2d3748;
        font-family: 'Poppins', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Section Headers */
    .param-section-header {
        color: #667eea;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Input Styling */
    .stSlider label, .stSelectbox label {
        color: #4a5568 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    .stSlider [data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .stSlider [data-baseweb="slider"] > div > div > div {
        background: #667eea !important;
    }
    
    /* Selectbox Styling */
    div[data-baseweb="select"] {
        border-radius: 10px;
    }
    
    div[data-baseweb="select"] > div {
        background-color: #f7fafc !important;
        border: 1px solid #e2e8f0 !important;
        color: #2d3748 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    div[data-baseweb="select"] * {
        color: #2d3748 !important;
    }
    
    /* Button Container */
    .button-container {
        text-align: center;
        margin: 2.5rem 0 0 0;
    }
    
    /* Main Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        padding: 0.9rem 3.5rem !important;
        border: none !important;
        border-radius: 50px !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
        width: auto !important;
        min-width: 350px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Results Card */
    .results-card {
        background: white;
        padding: 2rem 2.5rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        margin-top: 1rem;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        font-family: 'Poppins', sans-serif !important;
        color: #667eea !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #2d3748 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
    }
    
    /* Alert Styling */
    [data-testid="stSuccess"] {
        background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%) !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        color: #22543d !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stError"] {
        background: linear-gradient(135deg, #fed7d7 0%, #fc8181 100%) !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        color: #742a2a !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stWarning"] {
        background: linear-gradient(135deg, #feebc8 0%, #fbd38d 100%) !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        color: #7c2d12 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stInfo"] {
        background: linear-gradient(135deg, #bee3f8 0%, #90cdf4 100%) !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        color: #2c5282 !important;
        font-weight: 600 !important;
    }
    
    /* DataFrame */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Footer */
    .footer {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin-top: 2rem;
    }
    
    .footer p {
        margin: 0.5rem 0;
        color: #4a5568;
    }
    
    .footer strong {
        color: #667eea;
        font-weight: 700;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: #e2e8f0;
        margin: 1.5rem 0;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hide empty containers */
    .element-container:has(> .stMarkdown > div:empty) {
        display: none;
    }
    
    /* Remove extra spacing */
    .block-container > div:empty {
        display: none;
    }
    
    /* Reduce gap between button and results */
    .button-container + div {
        margin-top: -1rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .header-card h1 {
            font-size: 2rem;
        }
        
        .stButton > button {
            min-width: 250px;
            font-size: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    try:
        return load('model.pkl')
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run `model.py` first to train and save the model.")
        st.stop()

model = load_model()

# Header Section
st.markdown("""
<div class="header-card">
    <h1>🥛 Milk Quality Classifier</h1>
    <p>AI-Powered Quality Analysis System</p>
</div>
""", unsafe_allow_html=True)

# Input Parameters Card
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">📊 Input Parameters</h2>', unsafe_allow_html=True)

# Physical Properties Section
st.markdown('<div class="param-section-header">🔬 01. PHYSICAL PROPERTIES</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    ph = st.slider(
        "pH Level", 
        min_value=3.0, 
        max_value=9.5, 
        value=6.6, 
        step=0.1,
        help="Acidity/alkalinity level of milk (typical: 6.4-6.8)"
    )

with col2:
    temperature = st.slider(
        "Temperature (°C)", 
        min_value=20, 
        max_value=90, 
        value=45,
        help="Storage or processing temperature"
    )

col3, col4 = st.columns(2)
with col3:
    colour = st.slider(
        "Colour Value", 
        min_value=240, 
        max_value=260, 
        value=254,
        help="Color measurement indicating freshness"
    )

with col4:
    turbidity = st.selectbox(
        "Turbidity Level",
        options=[0, 1],
        help="Cloudiness or haziness level (0 = Low, 1 = High)"
    )

# Chemical Properties Section
st.markdown('<div class="param-section-header">🧪 02. CHEMICAL PROPERTIES</div>', unsafe_allow_html=True)

fat = st.selectbox(
    "Fat Content",
    options=[0, 1],
    help="Fat content level (0 = Not Optimal, 1 = Optimal)"
)

# Sensory Properties Section
st.markdown('<div class="param-section-header">👃 03. SENSORY PROPERTIES</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    taste = st.selectbox(
        "Taste Quality",
        options=[0, 1],
        help="Taste assessment (0 = Bad, 1 = Good)"
    )

with col6:
    odor = st.selectbox(
        "Odor Quality",
        options=[0, 1],
        help="Smell assessment (0 = Bad, 1 = Good)"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Create input DataFrame
input_data = pd.DataFrame({
    'ph': [ph],
    'temperature': [temperature],
    'taste': [taste],
    'odor': [odor],
    'fat': [fat],
    'turbidity': [turbidity],
    'colour': [colour]
})

# Centered Prediction Button
st.markdown('<div class="button-container">', unsafe_allow_html=True)
predict_button = st.button("🚀 Analyze Milk Quality", use_container_width=False, type="primary")
st.markdown('</div>', unsafe_allow_html=True)

# Display Results
if predict_button:
    with st.spinner("🔬 Analyzing milk sample..."):
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        grade_map = {0: "Low", 1: "Medium", 2: "High"}
        predicted_grade = grade_map[prediction]
        confidence = prediction_proba[prediction] * 100
        
        # Determine status and color
        if prediction == 2:
            status_text = "High Quality"
            status_color = "#10b981"  # Green
            description = "This milk sample meets excellent quality standards and is safe for consumption."
        elif prediction == 1:
            status_text = "Medium Quality"
            status_color = "#f59e0b"  # Orange
            description = "This milk sample is acceptable but could benefit from quality improvements."
        else:
            status_text = "Low Quality"
            status_color = "#ef4444"  # Red
            description = "This milk sample does not meet quality standards and requires immediate attention."
        
        # Results Section with Dark Card Style
        st.markdown("""
        <style>
        .dark-results-card {
            background: #000000;
            padding: 3rem 2.5rem;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
            margin: 2rem 0;
            text-align: center;
            border: 1px solid #333;
        }
        .results-header {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            margin-bottom: 2rem;
            color: #ffffff;
            font-size: 1.5rem;
            font-weight: 700;
        }
        .confidence-big {
            font-size: 5rem;
            font-weight: 800;
            color: #ffffff;
            margin: 1.5rem 0;
            line-height: 1;
        }
        .quality-status {
            font-size: 1.8rem;
            font-weight: 700;
            margin: 1.5rem 0;
        }
        .description-text {
            color: #e0e0e0;
            font-size: 1.1rem;
            margin: 2rem 0;
            line-height: 1.6;
        }
        .note-text {
            color: #b0b0b0;
            font-size: 0.9rem;
            font-style: italic;
            line-height: 1.6;
            margin-top: 2rem;
            padding-top: 2rem;
            border-top: 1px solid #444;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="dark-results-card">
            <div class="results-header">
                <span>📋</span>
                <span>Quality Analysis Complete</span>
                <span style="color: #94a3b8;">🔗</span>
            </div>
            <div class="confidence-big">{confidence:.1f}%</div>
            <div class="quality-status" style="color: {status_color};">Quality Level: {status_text}</div>
            <div class="description-text">{description}</div>
            <div class="note-text">
                <strong>Note:</strong> This is an ML estimation based on learned patterns from quality control data. 
                Model accuracy: 99.06%. Always follow standard laboratory testing procedures for official quality certification.
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p style="font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem;">
        Milk Quality Classifier | XGBoost Machine Learning
    </p>
    <p style="font-size: 0.9rem;">
        Created by <strong>Ayyapparaja</strong>
    </p>
    <p style="font-size: 0.85rem; color: #718096; margin-top: 0.5rem;">
        © 2026 | College Project
    </p>
</div>
""", unsafe_allow_html=True)
