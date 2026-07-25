"""
app.py - Dr. Friend main entry point
"""

import streamlit as st
import pandas as pd
from core.model_loader import load_models
from core.predictor import predict_disease
from core.urgency_engine import assess_urgency
from rag.disease_lookup import get_disease_overview
import os



st.set_page_config(
    page_title="Dr. Friend - AI Healthcare Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0d1117;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
    }

    [data-testid="stAppViewContainer"] > .main .block-container {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem 2rem;
    }
    .main .block-container {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem 2rem;
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 900px;
    }

    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #8b949e !important;
        font-size: 13px;
    }
    [data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
        font-weight: 600;
        font-size: 18px;
        letter-spacing: -0.3px;
    }
    [data-testid="stSidebar"] .stAlert {
        background: rgba(56, 189, 248, 0.1) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: #21262d !important;
        border: 1px solid #4d9fff !important;
        color: #ffffff !important;
        font-size: 13px !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #4d9fff !important;
        color: #ffffff !important;
    }

    h1 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        color: #e6edf3 !important;
        letter-spacing: -1px !important;
    }

    .stApp .stCaption {
        color: #8b949e !important;
        font-size: 14px !important;
        letter-spacing: 0.2px;
    }

    [data-testid="stChatMessage"] {
        background: #21262d !important;
        border-radius: 12px !important;
        padding: 8px 14px !important;
        margin-bottom: 8px !important;
        border: 1px solid #30363d !important;
    }

    [data-testid="stChatInput"] {
        border-radius: 12px !important;
        border: 2px solid #30363d !important;
        background: #161b22 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        transition: border-color 0.2s;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #0891b2 !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #e6edf3 !important;
        background: #161b22 !important;
        caret-color: #e6edf3 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #8b949e !important;
    }
    [data-testid="stChatInput"] > div {
        background: #161b22 !important;
    }

    [data-testid="stTextInput"] input {
        background: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }
    [data-testid="stTextInput"] input::placeholder {
        color: #8b949e !important;
    }
    [data-testid="stTextArea"] textarea {
        background: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }
    [data-testid="stTextArea"] textarea::placeholder {
        color: #8b949e !important;
    }

    .stAlert {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-left: 4px solid #0891b2 !important;
        border-radius: 10px !important;
        color: #e6edf3 !important;
        font-size: 14px !important;
    }

    hr {
        border-color: #30363d !important;
        margin: 1rem 0 !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
    }

    .stButton > button:not([kind="primary"]) {
        background: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        font-size: 14px !important;
    }

    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #0891b2, #0e7490) !important;
        border-radius: 4px !important;
    }
    .stProgress > div {
        background: #30363d !important;
        border-radius: 4px !important;
    }

    [data-testid="stDataFrame"] {
        border-radius: 10px !important;
        overflow: hidden !important;
        border: 1px solid #30363d !important;
    }

    [data-testid="stWarning"] {
        background: #2e2410 !important;
        border: 1px solid #f59e0b !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 8px !important;
        color: #fbbf24 !important;
        font-size: 14px !important;
    }
            
    [data-testid="stWarning"] {
        background: #2e2410 !important;
        border: 1px solid #f59e0b !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 8px !important;
        color: #fbbf24 !important;
        font-size: 14px !important;
    }

    [data-testid="stExpander"] summary {
        background: #161b22 !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary:hover {
        background: #21262d !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid #30363d !important;
        background: #0d1117 !important;
    }
            

    p, li {
        color: #e6edf3 !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
    }

    h2 {
        color: #e6edf3 !important;
        font-weight: 600 !important;
        font-size: 1.3rem !important;
        letter-spacing: -0.3px !important;
    }
    h3 {
        color: #e6edf3 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }

    .stSpinner > div {
        border-top-color: #0891b2 !important;
    }
</style>
""", unsafe_allow_html=True)

models = load_models()

if "stage" not in st.session_state:
    st.session_state.stage = "entry"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "urgency_result" not in st.session_state:
    st.session_state.urgency_result = None
if "emergency_vitals_requested" not in st.session_state:
    st.session_state.emergency_vitals_requested = False

# UC2-specific session state
if "uc2_messages" not in st.session_state:
    st.session_state.uc2_messages = []
if "uc2_result" not in st.session_state:
    st.session_state.uc2_result = None

# UC3-specific session state
if "uc3_result" not in st.session_state:
    st.session_state.uc3_result = None

ALL_STATE_KEYS = [
    "messages", "stage", "patient_data",
    "prediction_result", "urgency_result",
    "uc2_messages", "uc2_result",
    "uc3_result", "emergency_vitals_requested",
]  

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
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: #1e3a5f !important;
        border: 1px solid #4d9fff !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #4d9fff !important;
        border: 1px solid #4d9fff !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    if st.button("Reset Conversation", use_container_width=True):
        for key in ALL_STATE_KEYS:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    st.divider()
    st.caption("BITS Pilani WILP")
    st.caption("Bharath Krishna | 2024DA04347")
    st.divider()
    st.caption("Data: India Disease Burden (Lancet)")
    st.caption("Protocol: NHS NEWS2 Urgency")

st.markdown("""
<h1 style="font-family: Inter, sans-serif; font-weight: 700; 
font-size: 2.2rem; color: #0f172a; letter-spacing: -1px; margin-bottom: 0;">
Dr. <span style="color: #0891b2;">Friend</span>
</h1>
""", unsafe_allow_html=True)
st.caption("AI-Powered Healthcare Guidance and Triage Assistant")

# -----------------------------------------------------------------------
# DISEASE CATEGORY MAPPING
# Used in results screen for broad category label and self-care advice
# -----------------------------------------------------------------------
DISEASE_CATEGORY = {
    # Respiratory
    "Bronchitis": "Respiratory",
    "Pneumonia": "Respiratory",
    "Asthma": "Respiratory",
    "COPD": "Respiratory",
    "COVID-19": "Respiratory",
    "Lung Cancer": "Respiratory",
    "Tuberculosis": "Respiratory",
    # Cardiac
    "Arrhythmia": "Cardiac",
    "Heart Failure": "Cardiac",
    "Hypertensive Crisis": "Cardiac",
    # Metabolic
    "Type 2 Diabetes": "Metabolic",
    "Hypothyroidism": "Metabolic",
    "Anaemia": "Metabolic",
    # Infectious
    "Dengue Fever": "Infectious",
    "Typhoid": "Infectious",
    "Typhoid Fever": "Infectious",
    "Malaria": "Infectious",
    "Gastroenteritis": "Infectious",
    "Food Poisoning": "Infectious",
    "Hepatitis": "Infectious",
    "UTI": "Infectious",
    "Viral Infection": "Infectious",
    # Neurological
    "Migraine": "Neurological",
    "Anxiety Attack": "Neurological",
}

def render_doctor_discovery(disease_name: str, section_key: str):
    """
    Shared Doctor Discovery UI section, called identically from UC1,
    UC2, and UC3 results screens -- per the outline's design, this is
    a shared service, not rebuilt per use case.

    section_key must be unique per call site (e.g. "uc1", "uc2", "uc3")
    since Streamlit widgets need unique keys when the same component
    appears on multiple screens.
    """
    st.divider()
    st.subheader("Find Care Near You")

    from core.doctor_discovery import find_nearby_doctors, get_specialty_for_disease
    specialty = get_specialty_for_disease(disease_name)
    st.markdown(f"Based on your condition, a **{specialty}** would be the most relevant specialist.")

    location = st.text_input(
        "Enter your city or area",
        placeholder="e.g. Bandra, Mumbai",
        key=f"location_input_{section_key}",
    )

    if st.button("Find Doctors", key=f"find_doctors_btn_{section_key}"):
        if not location.strip():
            st.warning("Please enter a city or area to search.")
        else:
            with st.spinner(f"Finding {specialty}s near {location}..."):
                result = find_nearby_doctors(disease_name, location.strip())

            if result["error"]:
                st.error(f"Could not fetch results: {result['error']}")
            elif not result["results"]:
                st.info("No doctors found for this search. Try a nearby larger city or area.")
            else:
                for doc in result["results"]:
                    rating_display = f"⭐ {doc['rating']} ({doc['total_ratings']} reviews)" if doc["rating"] != "No rating" else "No rating yet"
                    st.markdown(f"""<div style="background: #21262d; border: 1px solid #30363d; border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;"><p style="font-weight:700; font-size:14.5px; color:#ffffff; margin-bottom:4px;">{doc['name']}</p><p style="font-size:12.5px; color:#4dd4e8; margin-bottom:4px;">{rating_display}</p><p style="font-size:12.5px; color:#8b949e; margin:0;">{doc['address']}</p></div>""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# SCREEN 0: ENTRY POINT
# -----------------------------------------------------------------------
if st.session_state.stage == "entry":
    st.markdown("""
    <style>
    .hero-wrap {
        text-align: center;
        padding: 8px 0 24px 0;
    }
    .hero-icon {
        font-size: 46px;
        margin-bottom: 4px;
    }
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 24px;
        color: #ffffff;
        margin-bottom: 6px;
    }
    .hero-sub {
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        color: #c9d1d9;
        max-width: 480px;
        margin: 0 auto;
        line-height: 1.6;
    }
    .entry-card {
        background: #21262d;
        border-radius: 18px;
        padding: 32px 20px 24px 20px;
        text-align: center;
        height: 100%;
        border: 1px solid #30363d;
        transition: all 0.28s ease;
        position: relative;
    }
    .entry-card:hover {
        transform: translateY(-6px);
        border-color: #67e8f9;
    }
    .entry-icon-circle {
        width: 76px;
        height: 76px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 18px auto;
        font-size: 36px;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.6);
    }
    .entry-icon-1 { background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); box-shadow: 0 8px 20px rgba(8, 145, 178, 0.35); }
    .entry-icon-2 { background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); box-shadow: 0 8px 20px rgba(34, 197, 94, 0.35); }
    .entry-icon-3 { background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%); box-shadow: 0 8px 20px rgba(168, 85, 247, 0.35); }
    .entry-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 17px;
        color: #ffffff;
        margin-bottom: 8px;
    }
    .entry-desc {
        font-family: 'Inter', sans-serif;
        font-size: 13px;
        color: #c9d1d9;
        line-height: 1.55;
        margin-bottom: 20px;
        min-height: 44px;
    }
    .step-strip {
        display: flex;
        justify-content: center;
        gap: 36px;
        margin-top: 28px;
        padding: 16px;
        flex-wrap: wrap;
    }
    .step-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'Inter', sans-serif;
        font-size: 12.5px;
        color: #c9d1d9;
    }
    .step-num {
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: #0891b2;
        color: white;
        font-size: 11px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    </style>

    <div class="hero-wrap">
        <div class="hero-icon">🩺</div>
        <div class="hero-title">Let's understand what you're feeling</div>
        <div class="hero-sub">Pick the option that matches what information you have right now. Dr. Friend will guide you through a short conversation and give you clear, sourced guidance.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="entry-card">
            <div class="entry-icon-circle entry-icon-1">🩺</div>
            <div class="entry-title">Vitals + Symptoms</div>
            <div class="entry-desc">Temperature, heart rate, blood pressure, or SpO2 available</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start with Vitals", type="primary", use_container_width=True, key="btn_uc1"):
            st.session_state.stage = "chat"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="entry-card">
            <div class="entry-icon-circle entry-icon-2">💬</div>
            <div class="entry-title">Symptoms Only</div>
            <div class="entry-desc">No measuring devices available - just describe how you feel</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start with Symptoms Only", type="primary", use_container_width=True, key="btn_uc2"):
            st.session_state.stage = "uc2_chat"
            st.rerun()

    with col3:
        st.markdown("""
        <div class="entry-card">
            <div class="entry-icon-circle entry-icon-3">📄</div>
            <div class="entry-title">Lab Report</div>
            <div class="entry-desc">Upload a PDF lab report and describe your symptoms</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Upload Lab Report", type="primary", use_container_width=True, key="btn_uc3"):
            st.session_state.stage = "uc3_upload"
            st.rerun()

    st.markdown("""
    <div class="step-strip">
        <div class="step-item"><div class="step-num">1</div>Choose your pathway</div>
        <div class="step-item"><div class="step-num">2</div>Answer a few quick questions</div>
        <div class="step-item"><div class="step-num">3</div>Get sourced, explainable guidance</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("🔒 Your information is used only for this session and is not stored permanently.")

# -----------------------------------------------------------------------
# SCREEN 1: CHAT (UC1)
# -----------------------------------------------------------------------
elif st.session_state.stage == "chat":
    if len(st.session_state.messages) == 0:
        
        st.markdown("""<div style="background: #161b22; border: 1px solid #30363d; border-left: 4px solid #0891b2; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px; font-size: 14px; color: #c9d1d9; font-family: Inter, sans-serif;">Dr. Friend will ask you a few questions about your symptoms and vitals. The conversation takes 2-3 minutes. Your information is used only for this session.</div>""", unsafe_allow_html=True)

    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🩺"):
                st.markdown(message["content"])
        else:
            with st.chat_message("user"):
                st.markdown(message["content"])

    if len(st.session_state.messages) == 0:
        opening = (
            "Hello! I am Dr. Friend. I am here to help you understand "
            "what might be going on with your health. "
            "Can you tell me how you have been feeling?"
        )
        with st.chat_message("assistant", avatar="🩺"):
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

        emergency_keywords = [
            "spo2 9", "oxygen 9", "o2 9", "saturation 9",
            "cant breathe", "can't breathe", "cannot breathe",
            "difficulty breathing", "not breathing",
            "bp 18", "blood pressure 18",
            "heart attack", "unconscious", "fainted", "passing out",
            "chest pain"
        ]
        user_lower = user_input.lower()
        is_emergency = any(keyword in user_lower for keyword in emergency_keywords)

        if is_emergency and len(st.session_state.messages) >= 2:
            if not st.session_state.emergency_vitals_requested:
                # SAFETY FIX (Bug 18): previously this jumped straight to
                # analysis on the FIRST emergency-keyword message, before
                # any vitals were ever collected. extract_patient_data()
                # then had nothing real to extract, silently substituting
                # default/normal-looking values -- which fed a genuinely
                # misleading input into the ML model (e.g. a real
                # emergency with SpO2 89% could get analysed as if SpO2
                # were a fabricated "98%"), and displayed those fake
                # values in the Vitals table as if they were measured.
                #
                # FIX: ask ONE consolidated follow-up for all vitals at
                # once, rather than looping through several separate
                # questions -- balancing genuine urgency (don't delay
                # emergency care with a long Q&A) against not analysing
                # on completely fabricated data. Whatever the patient
                # provides next, proceed to analysis regardless -- we do
                # NOT ask again a second time, since that would risk
                # delaying care further in a genuine emergency.
                emergency_reply = (
                    "This sounds urgent, and you should seek medical attention "
                    "right away. To help assess this as accurately as possible "
                    "in the meantime, if you have them available, please share "
                    "your temperature, heart rate, blood pressure, and SpO2 "
                    "(oxygen level) all together in one message. If you don't "
                    "have these, just let me know and I'll proceed with what "
                    "you've already told me."
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": emergency_reply}
                )
                with st.chat_message("assistant"):
                    st.markdown(emergency_reply)

                st.session_state.emergency_vitals_requested = True
                st.rerun()
            else:
                emergency_reply = (
                    "Thank you. I have enough information to analyse your "
                    "condition right away."
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": emergency_reply}
                )
                with st.chat_message("assistant"):
                    st.markdown(emergency_reply)

                conversation_text = ""
                for msg in st.session_state.messages:
                    role = "Patient" if msg["role"] == "user" else "Dr. Friend"
                    conversation_text += f"{role}: {msg['content']}\n\n"

                with st.spinner("Analysing emergency symptoms..."):
                    from llm.entity_extractor import extract_patient_data
                    st.session_state.patient_data = extract_patient_data(conversation_text)

                st.session_state.stage = "analysing"
                st.rerun()

        with st.spinner("Dr. Friend is thinking..."):
            from llm.conversation import get_dr_friend_response, is_ready_to_analyse, clean_response
            from llm.entity_extractor import extract_patient_data
            response = get_dr_friend_response(st.session_state.messages)
            cleaned = clean_response(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": cleaned}
        )
        with st.chat_message("assistant", avatar="🩺"):
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
# SCREEN 2: ANALYSING (UC1)
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
# SCREEN 3: RESULTS (UC1)
# -----------------------------------------------------------------------
elif st.session_state.stage == "results":
    prediction = st.session_state.prediction_result

    urgency = st.session_state.urgency_result
    top_disease = prediction["top_disease"]
    category = DISEASE_CATEGORY.get(top_disease, "General")

    urgency_styles = {
       "Self-Care at Home": {"dark_bg": "#14291b", "dark_border": "#22c55e", "dark_text": "#4ade80"},
        "See a Doctor Soon": {"dark_bg": "#2e2410", "dark_border": "#f59e0b", "dark_text": "#fbbf24"},
        "See a Doctor Today": {"dark_bg": "#2e1c0f", "dark_border": "#ea580c", "dark_text": "#fb923c"},
        "Go to Emergency": {"dark_bg": "#2e1414", "dark_border": "#dc2626", "dark_text": "#f87171"},
    }
    us = urgency_styles.get(urgency["level"], urgency_styles["See a Doctor Soon"])

    reasoning_html = "".join(
        f'<p style="margin:0 0 4px 0;">{rule}</p>' for rule in urgency["triggered_rules"]
    ) if urgency["triggered_rules"] else f'<p style="margin:0;">{urgency["description"]}</p>'

    ml_disease = prediction.get('ml_top_disease', prediction['top_disease'])
    ml_confidence = round(prediction.get('ml_confidence', prediction['top_confidence']) * 100, 1)
    layer2_status = "Activated" if prediction.get('layer2_triggered', False) else "Not triggered"
    layer2_color = "#dc2626" if prediction.get('layer2_triggered', False) else "#16a34a"
    final_source = prediction.get('prediction_source', 'ml_model').replace('_', ' ').title()

    st.markdown(f"""<div style="background: #1e1e1e; border: 1px solid #333333; border-radius: 16px; padding: 24px 26px; margin: 16px 0;">
<h3 style="margin: 0 0 14px 0; color: #ffffff;">Prediction transparency</h3>
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 22px;">
<div style="background: #2a2a2a; border-radius: 10px; padding: 12px 14px;">
<p style="font-size: 12px; color: #9ca3af; margin: 0 0 4px 0;">ML model output</p>
<p style="font-size: 14px; font-weight: 600; color: #ffffff; margin: 0;">{ml_disease} - {ml_confidence}%</p>
</div>
<div style="background: #2a2a2a; border-radius: 10px; padding: 12px 14px;">
<p style="font-size: 12px; color: #9ca3af; margin: 0 0 4px 0;">Layer 2 status</p>
<p style="font-size: 14px; font-weight: 600; color: {layer2_color}; margin: 0;">{layer2_status}</p>
</div>
<div style="background: #2a2a2a; border-radius: 10px; padding: 12px 14px;">
<p style="font-size: 12px; color: #9ca3af; margin: 0 0 4px 0;">Final decision</p>
<p style="font-size: 14px; font-weight: 600; color: #ffffff; margin: 0;">{top_disease}</p>
</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 22px;">
<div>
<h3 style="margin: 0 0 8px 0; color: #ffffff;">What might be going on</h3>
<p style="font-size: 24px; font-weight: 700; color: #ffffff; margin: 0 0 4px 0;">{top_disease}</p>
<p style="font-size: 13px; color: #4ade80; margin: 0;">{round(prediction['top_confidence'] * 100, 1)}% confidence</p>
<p style="font-size: 13px; color: #9ca3af; margin: 6px 0 0 0;">Broad category: {category}</p>
</div>
<div>
<h3 style="margin: 0 0 8px 0; color: #ffffff;">Urgency assessment</h3>
<p style="font-size: 18px; font-weight: 700; color: {us['dark_text']}; margin: 0 0 8px 0;">{urgency['level']}</p>
<div style="background: {us['dark_bg']}; border: 1px solid {us['dark_border']}; border-radius: 10px; padding: 12px 14px; color: {us['dark_text']}; font-size: 13.5px;">
{reasoning_html}
</div>
</div>
</div>""", unsafe_allow_html=True)

    from rag.disease_lookup import get_disease_overview
    overview = get_disease_overview(top_disease)
    if overview:
        st.markdown(f"""
        <h3 style="margin: 0 0 8px 0;">About {overview['disease_name']}</h3>
        <p style="font-size: 14px; color: #334155; margin: 0 0 4px 0;">{overview['overview_text']}</p>
        <p style="font-size: 12px; color: #94a3b8; margin: 0;">Source: {overview['citation']}</p>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

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

    render_doctor_discovery(top_disease, "uc1")

    if st.button("Start New Consultation", type="primary"):
        for key in ALL_STATE_KEYS:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    

# -----------------------------------------------------------------------
# SCREEN 4: CHAT (UC2 - Symptoms Only)
# -----------------------------------------------------------------------
elif st.session_state.stage == "uc2_chat":
    if len(st.session_state.uc2_messages) == 0:
        st.markdown("""<div style="background: #161b22; border: 1px solid #30363d; border-left: 4px solid #0891b2; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px; font-size: 14px; color: #c9d1d9; font-family: Inter, sans-serif;">Dr. Friend will ask about your symptoms only - no vitals or measuring devices needed. The conversation takes 1-2 minutes.</div>""", unsafe_allow_html=True)

    for message in st.session_state.uc2_messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🩺"):
                st.markdown(message["content"])
        else:
            with st.chat_message("user"):
                st.markdown(message["content"])

    if len(st.session_state.uc2_messages) == 0:
        opening = (
            "Hello! I am Dr. Friend. Since you don't have vitals available, "
            "just tell me about your symptoms and I'll ask a few follow-up "
            "questions. What symptoms have you been experiencing?"
        )
        with st.chat_message("assistant", avatar="🩺"):
            st.markdown(opening)
        st.session_state.uc2_messages.append(
            {"role": "assistant", "content": opening}
        )

    user_input = st.chat_input("Describe your symptoms...")

    if user_input:
        st.session_state.uc2_messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        from llm.uc2_conversation import get_uc2_response, check_readiness

        # Check readiness on the conversation as it stands right after
        # the patient's latest reply, BEFORE generating a new question.
        # This was previously checked AFTER generating a new assistant
        # response, which meant an unnecessary extra question always
        # got asked first -- see Bug 11/12 discussion. Checking here
        # instead means we can go straight to analysis the moment
        # enough information exists, without an unwanted extra turn.
        with st.spinner("Checking if I have enough information..."):
            ready = check_readiness(st.session_state.uc2_messages)

        if ready:
            st.session_state.stage = "uc2_analysing"
            st.rerun()
        else:
            with st.spinner("Dr. Friend is thinking..."):
                response = get_uc2_response(st.session_state.uc2_messages)

            st.session_state.uc2_messages.append(
                {"role": "assistant", "content": response}
            )
            with st.chat_message("assistant", avatar="🩺"):
                st.markdown(response)

# -----------------------------------------------------------------------
# SCREEN 5: ANALYSING (UC2)
# -----------------------------------------------------------------------
elif st.session_state.stage == "uc2_analysing":
    with st.spinner("Retrieving guidance and assessing urgency..."):
        try:
            from llm.uc2_conversation import run_uc2_pipeline
            result = run_uc2_pipeline(st.session_state.uc2_messages)
            st.session_state.uc2_result = result
            st.session_state.stage = "uc2_results"
            st.rerun()

        except Exception as e:
            st.error(f"UC2 analysis failed: {str(e)}")
            st.exception(e)
            if st.button("Go back"):
                st.session_state.stage = "uc2_chat"
                st.rerun()

# -----------------------------------------------------------------------
# SCREEN 6: RESULTS (UC2)
# -----------------------------------------------------------------------
elif st.session_state.stage == "uc2_results":
    result = result = st.session_state.uc2_result

    urgency_styles = {
        "Self-Care at Home": {"bg": "#14291b", "border": "#22c55e", "text": "#4ade80"},
        "See a Doctor Soon": {"bg": "#2e2410", "border": "#f59e0b", "text": "#fbbf24"},
        "See a Doctor Today": {"bg": "#2e1c0f", "border": "#ea580c", "text": "#fb923c"},
        "Go to Emergency": {"bg": "#2e1414", "border": "#dc2626", "text": "#f87171"},
    }
    us = urgency_styles.get(result["urgency_level"], urgency_styles["See a Doctor Soon"])
    primary_condition = result.get('primary_condition', 'Unclear')

    is_fallback = result.get("is_fallback_guess", False)
    condition_caption = (
        "General knowledge guess, not verified against our reference database - please treat as a starting point only"
        if is_fallback else
        "Based on retrieved guidance (no confidence score, not a classifier prediction)"
    )
    condition_caption_color = "#fbbf24" if is_fallback else "#9ca3af"

    st.markdown(f"""<div style="background: #1e1e1e; border: 1px solid #333333; border-radius: 16px; padding: 24px 26px; margin: 16px 0;"><h3 style="margin: 0 0 14px 0; color: #ffffff;">Retrieval transparency</h3><div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 22px;"><div style="background: #2a2a2a; border-radius: 10px; padding: 12px 14px;"><p style="font-size: 12px; color: #9ca3af; margin: 0 0 4px 0;">Retrieval method</p><p style="font-size: 14px; font-weight: 600; color: #ffffff; margin: 0;">RAG, {result['chunks_retrieved']} chunks used</p></div><div style="background: #2a2a2a; border-radius: 10px; padding: 12px 14px;"><p style="font-size: 12px; color: #9ca3af; margin: 0 0 4px 0;">Sources consulted</p><p style="font-size: 14px; font-weight: 600; color: #ffffff; margin: 0;">{len(result['sources'])} document(s)</p></div><div style="background: #2a2a2a; border-radius: 10px; padding: 12px 14px;"><p style="font-size: 12px; color: #9ca3af; margin: 0 0 4px 0;">Pathway</p><p style="font-size: 14px; font-weight: 600; color: #ffffff; margin: 0;">UC2 - symptoms only</p></div></div><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 22px;"><div><h3 style="margin: 0 0 8px 0; color: #ffffff;">What might be going on</h3><p style="font-size: 24px; font-weight: 700; color: #ffffff; margin: 0 0 4px 0;">{primary_condition}</p><p style="font-size: 13px; color: {condition_caption_color}; margin: 0;">{condition_caption}</p></div><div><h3 style="margin: 0 0 8px 0; color: #ffffff;">Urgency assessment</h3><p style="font-size: 18px; font-weight: 700; color: {us['text']}; margin: 0 0 8px 0;">{result['urgency_level']}</p><div style="background: {us['bg']}; border: 1px solid {us['border']}; border-radius: 10px; padding: 12px 14px; color: {us['text']}; font-size: 13.5px;"><p style="margin:0;">{result['urgency_reasoning']}</p></div></div></div></div>""", unsafe_allow_html=True)

    if result["urgency_matched_criteria"]:  
        with st.expander("Why this urgency level was flagged", expanded=True):
            for criterion in result["urgency_matched_criteria"]:
                st.warning(criterion)

    overview = get_disease_overview(result["primary_condition"]) if result.get("primary_condition") else None
    if overview:
        st.markdown(f"""<div style="background: #1e1e1e; border: 1px solid #333333; border-radius: 16px; padding: 20px 22px; margin-bottom: 16px;"><h3 style="margin: 0 0 8px 0; color: #ffffff;">About {overview['disease_name']}</h3><p style="font-size: 14px; color: #c9d1d9; margin: 0 0 4px 0;">{overview['overview_text']}</p><p style="font-size: 12px; color: #9ca3af; margin: 0;">Source: {overview['citation']}</p></div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style="background: #1e1e1e; border-left: 4px solid #0891b2; border-radius: 12px; padding: 20px 22px; margin-bottom: 16px;"><p style="color: #ffffff; font-weight: 700; font-size: 14px; margin-bottom: 8px;">Guidance</p><p style="color: #c9d1d9; font-size: 14.5px; line-height: 1.65; margin: 0;">{result['guidance']}</p></div>""", unsafe_allow_html=True)

    
    with st.expander("What Dr. Friend understood from your description"):
        st.markdown(result["symptom_summary"])

    with st.expander("How was this generated? (sources and retrieval details)"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Retrieval Method**")
            st.markdown("Source: RAG (LangChain + ChromaDB)")
            st.markdown(f"Chunks used: {result['chunks_retrieved']}")
        with col2:
            st.markdown("**Sources Consulted**")
            for source in result["sources"]:
                st.markdown(f"- {source}")
        if result.get("source_excerpts"):
            st.markdown("**Retrieved source text:**")
            for excerpt in result["source_excerpts"]:
                st.markdown(f"**{excerpt['citation']}** - *{excerpt['section']}*")
                st.markdown(f"> {excerpt['text']}")
                st.markdown("---")

    st.divider()
    st.caption(
        "Dr. Friend is a healthcare guidance assistant, not a replacement "
        "for professional medical advice. Always consult a qualified doctor."
    )

    render_doctor_discovery(result.get("primary_condition", "General"), "uc2")

    if st.button("Start New Consultation", type="primary"):
        for key in ALL_STATE_KEYS:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # -----------------------------------------------------------------------
# SCREEN 7: UPLOAD (UC3)
# -----------------------------------------------------------------------
elif st.session_state.stage == "uc3_upload":
    st.markdown("""
    <div style="
            background: #161b22;
            border: 1px solid #30363d;
            border-left: 4px solid #0891b2;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 16px;
            font-size: 14px;
            color: #c9d1d9;
            font-family: Inter, sans-serif;
        ">
        Upload your lab report (PDF) and briefly describe your symptoms.
        Dr. Friend will extract the values, flag anything abnormal, and
        combine this with your symptoms for guidance.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload lab report (PDF)", type=["pdf"])
    symptoms_input = st.text_area(
        "Describe your symptoms",
        placeholder="e.g. I've been feeling very tired and weak lately, and get dizzy when I stand up quickly",
        height=100,
    )

    if st.button("Analyse Report", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.error("Please upload a lab report PDF before continuing.")
        elif not symptoms_input.strip():
            st.error("Please describe your symptoms before continuing.")
        else:
            temp_path = os.path.join("data", "sample_report", "_uploaded_temp.pdf")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.uc3_pdf_path = temp_path
            st.session_state.uc3_symptoms = symptoms_input.strip()
            st.session_state.stage = "uc3_analysing"
            st.rerun()

# -----------------------------------------------------------------------
# SCREEN 8: ANALYSING (UC3)
# -----------------------------------------------------------------------
elif st.session_state.stage == "uc3_analysing":
    with st.spinner("Extracting values, flagging abnormal results, and retrieving guidance..."):
        try:
            from core.uc3_extractor import extract_structured_values
            from core.uc3_interpreter import interpret_lab_report

            structured_values = extract_structured_values(st.session_state.uc3_pdf_path)

            if not structured_values:
                st.error(
                    "No lab values could be extracted from this PDF. It may be a "
                    "scanned image rather than a digital PDF, which is not yet supported."
                )
                if st.button("Go back"):
                    st.session_state.stage = "uc3_upload"
                    st.rerun()
            else:
                result = interpret_lab_report(structured_values, st.session_state.uc3_symptoms)
                result["all_values"] = structured_values
                st.session_state.uc3_result = result
                st.session_state.stage = "uc3_results"
                st.rerun()

        except Exception as e:
            st.error(f"UC3 analysis failed: {str(e)}")
            st.exception(e)
            if st.button("Go back"):
                st.session_state.stage = "uc3_upload"
                st.rerun()

# -----------------------------------------------------------------------
# SCREEN 9: RESULTS (UC3)
# -----------------------------------------------------------------------
elif st.session_state.stage == "uc3_results":
    result = st.session_state.uc3_result

    st.subheader("Extracted Lab Values")
    all_values_data = {
        "Test Name": [v["test_name"] for v in result["all_values"]],
        "Result": [v["raw_result"] for v in result["all_values"]],
        "Reference Range": [v["raw_range"] for v in result["all_values"]],
        "Flag": [v["computed_flag"] for v in result["all_values"]],
    }
    st.dataframe(pd.DataFrame(all_values_data), hide_index=True, use_container_width=True)

    urgency_styles = {
        "Self-Care at Home": {"bg": "#14291b", "border": "#22c55e", "text": "#4ade80"},
        "See a Doctor Soon": {"bg": "#2e2410", "border": "#f59e0b", "text": "#fbbf24"},
        "See a Doctor Today": {"bg": "#2e1c0f", "border": "#ea580c", "text": "#fb923c"},
        "Go to Emergency": {"bg": "#2e1414", "border": "#dc2626", "text": "#f87171"},
    }
    us = urgency_styles.get(result["urgency_level"], urgency_styles["See a Doctor Soon"])
    primary_condition = result.get("primary_condition", "Unclear")

    st.markdown(f"""<div style="background: #1e1e1e; border: 1px solid #333333; border-radius: 16px; padding: 24px 26px; margin: 16px 0;">
<h3 style="margin: 0 0 14px 0; color: #ffffff;">Retrieval transparency</h3>
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 22px;">
<div style="background: #2a2a2a; border-radius: 10px; padding: 12px 14px;">
<p style="font-size: 12px; color: #9ca3af; margin: 0 0 4px 0;">Retrieval method</p>
<p style="font-size: 14px; font-weight: 600; color: #ffffff; margin: 0;">RAG, {result['chunks_retrieved']} chunks used</p>
</div>
<div style="background: #2a2a2a; border-radius: 10px; padding: 12px 14px;">
<p style="font-size: 12px; color: #9ca3af; margin: 0 0 4px 0;">Sources consulted</p>
<p style="font-size: 14px; font-weight: 600; color: #ffffff; margin: 0;">{len(result['sources'])} document(s)</p>
</div>
<div style="background: #2a2a2a; border-radius: 10px; padding: 12px 14px;">
<p style="font-size: 12px; color: #9ca3af; margin: 0 0 4px 0;">Pathway</p>
<p style="font-size: 14px; font-weight: 600; color: #ffffff; margin: 0;">UC3 - lab report, {len(result['abnormal_findings'])} abnormal</p>
</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 22px;">
<div>
<h3 style="margin: 0 0 8px 0; color: #ffffff;">What might be going on</h3>
<p style="font-size: 24px; font-weight: 700; color: #ffffff; margin: 0 0 4px 0;">{primary_condition}</p>
<p style="font-size: 13px; color: #9ca3af; margin: 0;">Based on lab findings and symptoms (no confidence score, not a classifier prediction)</p>
</div>
<div>
<h3 style="margin: 0 0 8px 0; color: #ffffff;">Urgency assessment</h3>
<p style="font-size: 18px; font-weight: 700; color: {us['text']}; margin: 0 0 8px 0;">{result['urgency_level']}</p>
<div style="background: {us['bg']}; border: 1px solid {us['border']}; border-radius: 10px; padding: 12px 14px; color: {us['text']}; font-size: 13.5px;">
<p style="margin:0;">{result['urgency_reasoning']}</p>
</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    if result["urgency_matched_criteria"]:
        with st.expander("Why this urgency level was flagged", expanded=True):
            for criterion in result["urgency_matched_criteria"]:
                st.warning(criterion)

    overview = get_disease_overview(primary_condition) if primary_condition and primary_condition != "Unclear" else None
    if overview:
        st.markdown(f"""<div style="background: #1e1e1e; border: 1px solid #333333; border-radius: 16px; padding: 20px 22px; margin-bottom: 16px;">
<h3 style="margin: 0 0 8px 0; color: #ffffff;">About {overview['disease_name']}</h3>
<p style="font-size: 14px; color: #c9d1d9; margin: 0 0 4px 0;">{overview['overview_text']}</p>
<p style="font-size: 12px; color: #9ca3af; margin: 0;">Source: {overview['citation']}</p>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style="background: #1e1e1e; border-left: 4px solid #0891b2; border-radius: 12px; padding: 20px 22px; margin-bottom: 16px;">
<p style="color: #ffffff; font-weight: 700; font-size: 14px; margin-bottom: 8px;">Interpretation</p>
<p style="color: #c9d1d9; font-size: 14.5px; line-height: 1.65; margin: 0;">{result['interpretation']}</p>
</div>""", unsafe_allow_html=True)

    if result["sources"]:
        with st.expander("How was this generated? (sources and retrieval details)"):
            st.markdown("**Sources Consulted**")
            for source in result["sources"]:
                st.markdown(f"- {source}")
            if result.get("source_excerpts"):
                st.markdown("**Retrieved source text:**")
                for excerpt in result["source_excerpts"]:
                    st.markdown(f"**{excerpt['citation']}** - *{excerpt['section']}*")
                    st.markdown(f"> {excerpt['text']}")
                    st.markdown("---")

    st.divider()
    st.caption(
        "Dr. Friend is a healthcare guidance assistant, not a replacement "
        "for professional medical advice. Always consult a qualified doctor."
    )

    render_doctor_discovery(result.get("primary_condition", "General"), "uc3")

    if st.button("Start New Consultation", type="primary"):
        for key in ALL_STATE_KEYS:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    