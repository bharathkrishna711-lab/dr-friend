"""
app.py - Dr. Friend main entry point
"""

import streamlit as st
import pandas as pd
from core.model_loader import load_models
from core.predictor import predict_disease
from core.urgency_engine import assess_urgency




st.set_page_config(
    page_title="Dr. Friend - AI Healthcare Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f0f4f8;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: none;
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        color: #94a3b8 !important;
        font-size: 13px;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
        font-weight: 600;
        font-size: 18px;
        letter-spacing: -0.3px;
    }
    
    /* Sidebar success box */
    [data-testid="stSidebar"] .stAlert {
        background: rgba(56, 189, 248, 0.1) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Sidebar button */
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        border: 1px solid #334155 !important;
        color: #94a3b8 !important;
        font-size: 13px !important;
        border-radius: 8px !important;
        transition: all 0.2s;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
        background: rgba(56, 189, 248, 0.05) !important;
    }
    
    /* Title */
    h1 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        color: #0f172a !important;
        letter-spacing: -1px !important;
    }
    
    /* Subtitle caption */
    .stApp .stCaption {
        color: #64748b !important;
        font-size: 14px !important;
        letter-spacing: 0.2px;
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 4px 8px !important;
        margin-bottom: 8px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    
    /* Chat input */
    [data-testid="stChatInput"] {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        background: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
        transition: border-color 0.2s;
    }
    
    [data-testid="stChatInput"]:focus-within {
        border-color: #0891b2 !important;
    }
    
    /* Info box */
    .stAlert {
        background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%) !important;
        border: 1px solid #bae6fd !important;
        border-left: 4px solid #0891b2 !important;
        border-radius: 10px !important;
        color: #0c4a6e !important;
        font-size: 14px !important;
    }
    
    /* Divider */
    hr {
        border-color: #e2e8f0 !important;
        margin: 1rem 0 !important;
    }
    
    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 4px 12px rgba(8, 145, 178, 0.3) !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 16px rgba(8, 145, 178, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Secondary buttons */
    .stButton > button:not([kind="primary"]) {
        background: white !important;
        color: #475569 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        font-size: 14px !important;
    }
    
    /* Metrics */
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #0891b2, #0e7490) !important;
        border-radius: 4px !important;
    }
    
    .stProgress > div {
        background: #e2e8f0 !important;
        border-radius: 4px !important;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px !important;
        overflow: hidden !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Warning boxes (urgency rules) */
    [data-testid="stWarning"] {
        background: #fffbeb !important;
        border: 1px solid #fcd34d !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 8px !important;
        color: #78350f !important;
        font-size: 14px !important;
    }
    
    /* General text */
    p, li {
        color: #334155 !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
    }
    
    h2 {
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 1.3rem !important;
        letter-spacing: -0.3px !important;
    }
    
    h3 {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #0891b2 !important;
    }
</style>
""", unsafe_allow_html=True)

models = load_models()

if "stage" not in st.session_state:
    st.session_state.stage = "chat"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "urgency_result" not in st.session_state:
    st.session_state.urgency_result = None

with st.sidebar:
    st.markdown("### Dr. Friend")
    st.markdown("AI Healthcare Guidance Assistant")
    st.divider()
    if models["loaded"]:
        st.success("Model Ready")
        st.caption(f"Model: {models['model_info'].get('model_name', 'Random Forest')}")
        st.caption(f"Version: {models['model_info'].get('version', 'v2.0')}")
        st.caption(f"Accuracy: {models['model_info'].get('accuracy', 'N/A')}%")
    else:
        st.error("Model failed to load")
        st.caption(models.get("error", "Unknown error"))
    st.divider()
    if st.button("Reset Conversation", use_container_width=True):
        for key in ["messages", "stage", "patient_data",
                    "prediction_result", "urgency_result"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    st.divider()
    st.caption("BITS Pilani WILP")
    st.caption("Bharath Krishna | 2024DA04347")

st.markdown("""
<h1 style="font-family: Inter, sans-serif; font-weight: 700; 
font-size: 2.2rem; color: #0f172a; letter-spacing: -1px; margin-bottom: 0;">
Dr. <span style="color: #0891b2;">Friend</span>
</h1>
""", unsafe_allow_html=True)
st.caption("AI-Powered Healthcare Guidance and Triage Assistant")
st.divider()

# -----------------------------------------------------------------------
# DISEASE CATEGORY MAPPING
# Used in results screen for broad category label and self-care advice
# -----------------------------------------------------------------------
DISEASE_CATEGORY = {
    "Bronchitis": "Respiratory", "Pneumonia": "Respiratory",
    "Asthma": "Respiratory", "COPD": "Respiratory",
    "COVID-19": "Respiratory", "Lung Cancer": "Respiratory",
    "Tuberculosis": "Respiratory",
    "Dengue Fever": "Infectious", "Typhoid": "Infectious",
    "Hypothyroidism": "Metabolic", "Type 2 Diabetes": "Metabolic",
    "Hypertensive Crisis": "Metabolic",
    "Migraine": "Neurological", "Anxiety Attack": "Neurological",
    "Anaemia": "Metabolic", "Arrhythmia": "Respiratory",
    "Heart Failure": "Respiratory"
}

SELF_CARE_ADVICE = {
    "Respiratory": [
        "Rest and avoid physical exertion",
        "Drink plenty of warm fluids (water, soups, herbal tea)",
        "Steam inhalation can ease chest tightness",
        "Keep checking your SpO2 every few hours",
        "If oxygen drops below 93% go to emergency immediately",
        "Avoid cold environments and stay warm"
    ],
    "Infectious": [
        "Rest and stay hydrated",
        "Monitor your temperature every few hours",
        "Avoid contact with others to prevent spread",
        "Eat light, easily digestible foods",
        "Take paracetamol for fever if needed"
    ],
    "Metabolic": [
        "Monitor your blood sugar levels regularly",
        "Stick to a low sugar, balanced diet",
        "Stay hydrated with water",
        "Avoid skipping meals",
        "Keep a record of your readings to show your doctor"
    ],
    "Neurological": [
        "Rest in a quiet, dark room",
        "Stay hydrated",
        "Avoid screen time if you have a headache",
        "Note when symptoms started and their severity",
        "Avoid triggers like bright lights or loud sounds"
    ]
}

# -----------------------------------------------------------------------
# SCREEN 1: CHAT
# -----------------------------------------------------------------------
if st.session_state.stage == "chat":

    # Welcome card - only shown before conversation starts
    if len(st.session_state.messages) == 0:
        if len(st.session_state.messages) == 0:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
                border: 1px solid #bae6fd;
                border-left: 4px solid #0891b2;
                border-radius: 10px;
                padding: 16px 20px;
                margin-bottom: 16px;
                font-size: 14px;
                color: #0c4a6e;
                font-family: Inter, sans-serif;
            ">
                Dr. Friend will ask you a few questions about your symptoms and vitals. 
                The conversation takes 2-3 minutes. 
                Your information is used only for this session.
            </div>
            """, unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if len(st.session_state.messages) == 0:
        opening = (
            "Hello! I am Dr. Friend. I am here to help you understand "
            "what might be going on with your health. "
            "Can you tell me how you have been feeling?"
        )
        with st.chat_message("assistant"):
            st.markdown(opening)
        st.session_state.messages.append(
            {"role": "assistant", "content": opening}
        )

    user_input = st.chat_input("Describe how you are feeling...")

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Dr. Friend is thinking..."):
            from llm.conversation import get_dr_friend_response, is_ready_to_analyse, clean_response
            from llm.entity_extractor import extract_patient_data
            response = get_dr_friend_response(st.session_state.messages)
            cleaned = clean_response(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": cleaned}
        )
        with st.chat_message("assistant"):
            st.markdown(cleaned)

        if is_ready_to_analyse(response):
            conversation_text = ""
            for msg in st.session_state.messages:
                role = "Patient" if msg["role"] == "user" else "Dr. Friend"
                conversation_text += f"{role}: {msg['content']}\n\n"

            with st.spinner("Extracting health information..."):
                st.session_state.patient_data = extract_patient_data(conversation_text)

            st.session_state.stage = "analysing"
            st.rerun()

# -----------------------------------------------------------------------
# SCREEN 2: ANALYSING
# -----------------------------------------------------------------------
elif st.session_state.stage == "analysing":
    with st.spinner("Analysing your symptoms..."):
        try:
            from core.prediction_agent import predict_with_agent
            prediction = predict_with_agent(
                patient_data=st.session_state.patient_data,
                model=models["model"],
                scaler=models["scaler"],
                label_encoder=models["label_encoder"]
            )
            urgency = assess_urgency(
                vitals=st.session_state.patient_data,
                predicted_disease=prediction["top_disease"]
            )
            st.session_state.prediction_result = prediction
            st.session_state.urgency_result = urgency
            st.session_state.stage = "results"
            st.rerun()

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)
            if st.button("Go back"):
                st.session_state.stage = "chat"
                st.rerun()

# -----------------------------------------------------------------------
# SCREEN 3: RESULTS
# -----------------------------------------------------------------------
elif st.session_state.stage == "results":
    prediction = st.session_state.prediction_result
    urgency = st.session_state.urgency_result
    top_disease = prediction["top_disease"]
    category = DISEASE_CATEGORY.get(top_disease, "General")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("What Might Be Going On")
        st.metric(
            label="Most Likely Condition",
            value=top_disease,
            delta=f"{round(prediction['top_confidence'] * 100, 1)}% confidence"
        )
        st.markdown(f"**Broad Category:** {category}")
        st.markdown("**Top 5 Possibilities**")
        for disease, prob in prediction["top_5"]:
            st.progress(prob, text=f"{disease}: {round(prob*100, 1)}%")

    with col_right:
        st.subheader("Urgency Assessment")
        urgency_colors = {
            "Self-Care at Home": "green",
            "See a Doctor Soon": "orange",
            "See a Doctor Today": "red",
            "Go to Emergency": "darkred"
        }
        color = urgency_colors.get(urgency["level"], "gray")
        st.markdown(
            f"<h2 style='color:{color}'>{urgency['level']}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(urgency["description"])
        if urgency["triggered_rules"]:
            st.markdown("**Why we flagged this:**")
            for rule in urgency["triggered_rules"]:
                st.warning(rule)

    # Vitals table
    st.divider()
    st.subheader("Your Vitals at a Glance")
    patient = st.session_state.patient_data
    vitals_data = {
        "Vital Sign": ["Temperature", "Heart Rate", "Blood Pressure", "SpO2", "BMI"],
        "Your Value": [
            f"{patient.get('temperature_c', 'N/A')}°C",
            f"{patient.get('heart_rate_bpm', 'N/A')} bpm",
            f"{patient.get('systolic_bp', 'N/A')}/{patient.get('diastolic_bp', 'N/A')} mmHg",
            f"{patient.get('spo2_percent', 'N/A')}%",
            f"{patient.get('bmi', 'N/A')}"
        ],
        "Normal Range": [
            "36.1 - 37.2°C",
            "60 - 100 bpm",
            "< 120/80 mmHg",
            "96 - 100%",
            "18.5 - 24.9"
        ]
    }
    st.dataframe(
        pd.DataFrame(vitals_data),
        hide_index=True,
        use_container_width=True
    )

    # Self-care advice - personalized using LLM
    st.divider()
    st.subheader("What You Can Do Right Now")
    from llm.conversation import generate_self_care_advice
    with st.spinner("Generating personalized advice..."):
        advice_list = generate_self_care_advice(
            top_disease,
            st.session_state.patient_data
        )
    for advice in advice_list:
        st.markdown(f"- {advice}")

    st.divider()
    st.caption(
        "Dr. Friend is a healthcare guidance assistant, not a replacement "
        "for professional medical advice. Always consult a qualified doctor."
    )

    with st.expander("Debug: Extracted Patient Data"):
        st.json(st.session_state.patient_data)

    if st.button("Start New Consultation", type="primary"):
        for key in ["messages", "stage", "patient_data",
                    "prediction_result", "urgency_result"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()