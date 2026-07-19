"""
app.py - Dr. Friend main entry point
"""

import streamlit as st
import pandas as pd
from core.model_loader import load_models
from core.predictor import predict_disease
from core.urgency_engine import assess_urgency
import os



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
            
    /* Subtle dot pattern background */
    [data-testid="stAppViewContainer"] {
        background-image: radial-gradient(#c8d8e8 1px, transparent 1px);
        background-size: 24px 24px;
        background-color: #f0f4f8;
    }
    
    /* Content area white card */
    [data-testid="stAppViewContainer"] > .main .block-container {
        background: rgba(255, 255, 255, 0.88);
        border-radius: 12px;
        padding: 1rem 2rem;
    }
    .main .block-container {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 12px;
        padding: 1rem 2rem;
    }        
    
    /* Main content area */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
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
    
    [data-testid="stSidebar"] .stButton > button {
        background: #1e3a5f !important;
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
        background: #dcf8c6 !important;
        border-radius: 12px !important;
        padding: 8px 14px !important;
        margin-bottom: 8px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
        border: none !important;
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
    st.session_state.stage = "entry"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "urgency_result" not in st.session_state:
    st.session_state.urgency_result = None

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
    "uc3_result",
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

# -----------------------------------------------------------------------
# SCREEN 0: ENTRY POINT
# -----------------------------------------------------------------------
if st.session_state.stage == "entry":
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
        border: 1px solid #bae6fd;
        border-left: 4px solid #0891b2;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 24px;
        font-size: 14px;
        color: #0c4a6e;
        font-family: Inter, sans-serif;
    ">
        Choose how you'd like to describe your health concern.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### I have vitals + symptoms")
        st.caption("Temperature, heart rate, blood pressure, or SpO2 available")
        if st.button("Start with Vitals", type="primary", use_container_width=True):
            st.session_state.stage = "chat"
            st.rerun()
    with col2:
        st.markdown("### I only know my symptoms")
        st.caption("No measuring devices available - symptoms only")
        if st.button("Start with Symptoms Only", type="primary", use_container_width=True):
            st.session_state.stage = "uc2_chat"
            st.rerun()
    with col3:
        st.markdown("### I have a lab report")
        st.caption("Upload a PDF lab report and describe your symptoms")
        if st.button("Upload Lab Report", type="primary", use_container_width=True):
            st.session_state.stage = "uc3_upload"
            st.rerun()

# -----------------------------------------------------------------------
# SCREEN 1: CHAT (UC1)
# -----------------------------------------------------------------------
elif st.session_state.stage == "chat":
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
            emergency_reply = (
                "These symptoms need immediate medical attention. "
                "I have enough information to analyse your condition right away."
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

    st.divider()
    st.subheader("Prediction Transparency")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**ML Model Output**")
        st.markdown(f"Disease: **{prediction.get('ml_top_disease', prediction['top_disease'])}**")
        st.markdown(f"Confidence: **{round(prediction.get('ml_confidence', prediction['top_confidence'])*100, 1)}%**")
        model_name = models['model_info'].get('model_name', 'XGBoost Tuned1')
        model_version = models['model_info'].get('version', 'v3.0')
        st.markdown(f"Source: {model_name} {model_version}")
    with col2:
        st.markdown("**Layer 2 Status**")
        if prediction.get('layer2_triggered', False):
            st.error("Activated")
            st.markdown("Low confidence or category mismatch detected")
        else:
            st.success("Not triggered")
            st.markdown("ML prediction was confident and accurate")

    with col3:
        st.markdown("**Final Decision**")
        st.markdown(f"Disease: **{prediction['top_disease']}**")
        st.markdown(f"Source: **{prediction.get('prediction_source', 'ml_model').replace('_', ' ').title()}**")
        if prediction.get('reasoning'):
            st.markdown(f"*{prediction['reasoning']}*")

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
            pct = round(prob * 100, 1)
            if pct >= 50:
                bar_color = "#0891b2"
            elif pct >= 20:
                bar_color = "#0e7490"
            else:
                bar_color = "#94a3b8"
            st.markdown(f"""
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; 
                            margin-bottom: 3px;">
                    <span style="font-size: 13px; font-weight: 500; 
                                color: #1e293b;">{disease}</span>
                    <span style="font-size: 13px; font-weight: 600; 
                                color: {bar_color};">{pct}%</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 6px; 
                            height: 10px; width: 100%;">
                    <div style="background: {bar_color}; border-radius: 6px; 
                                height: 10px; width: {max(pct, 2)}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

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
            Dr. Friend will ask about your symptoms only - no vitals or
            measuring devices needed. The conversation takes 1-2 minutes.
        </div>
        """, unsafe_allow_html=True)

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

        with st.spinner("Dr. Friend is thinking..."):
            from llm.uc2_conversation import get_uc2_response, check_readiness
            response = get_uc2_response(st.session_state.uc2_messages)

        st.session_state.uc2_messages.append(
            {"role": "assistant", "content": response}
        )
        with st.chat_message("assistant", avatar="🩺"):
            st.markdown(response)

        with st.spinner("Checking if I have enough information..."):
            ready = check_readiness(st.session_state.uc2_messages)

        if ready:
            st.session_state.stage = "uc2_analysing"
            st.rerun()

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
    result = st.session_state.uc2_result

    st.divider()
    st.subheader("Retrieval Transparency")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Retrieval Method**")
        st.markdown("Source: **RAG (LangChain + ChromaDB)**")
        st.markdown(f"Chunks used: **{result['chunks_retrieved']}**")

    with col2:
        st.markdown("**Sources Consulted**")
        for source in result["sources"]:
            st.markdown(f"- {source}")
        with st.expander("View retrieved source text"):
            for excerpt in result["source_excerpts"]:
                st.markdown(f"**{excerpt['citation']}** — *{excerpt['section']}*")
                st.markdown(f"> {excerpt['text']}")
                st.markdown("---")

    with col3:
        st.markdown("**Pathway**")
        st.markdown("Use Case: **UC2 - Symptoms Only**")
        st.markdown("*No vitals required or collected*")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("What Might Be Going On")
        st.markdown("**Symptom Summary**")
        st.markdown(f"*{result['symptom_summary']}*")
        st.markdown("**Guidance**")
        st.markdown(result["guidance"])

    with col_right:
        st.subheader("Urgency Assessment")
        urgency_colors = {
            "Self-Care at Home": "green",
            "See a Doctor Soon": "orange",
            "See a Doctor Today": "red",
            "Go to Emergency": "darkred"
        }
        color = urgency_colors.get(result["urgency_level"], "gray")
        st.markdown(
            f"<h2 style='color:{color}'>{result['urgency_level']}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(f"*{result['urgency_reasoning']}*")
        if result["urgency_matched_criteria"]:
            st.markdown("**Why we flagged this:**")
            for criterion in result["urgency_matched_criteria"]:
                st.warning(criterion)
        else:
            st.info("No specific red-flag criteria matched.")

    st.divider()
    st.caption(
        "Dr. Friend is a healthcare guidance assistant, not a replacement "
        "for professional medical advice. Always consult a qualified doctor."
    )

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

    st.divider()
    st.subheader("Extracted Lab Values")

    all_values_data = {
        "Test Name": [v["test_name"] for v in result["all_values"]],
        "Result": [v["raw_result"] for v in result["all_values"]],
        "Reference Range": [v["raw_range"] for v in result["all_values"]],
        "Flag": [v["computed_flag"] for v in result["all_values"]],
    }
    st.dataframe(pd.DataFrame(all_values_data), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Retrieval Transparency")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Retrieval Method**")
        st.markdown("Source: **RAG (LangChain + ChromaDB)**")
        st.markdown(f"Chunks used: **{result['chunks_retrieved']}**")
    with col2:
        st.markdown("**Sources Consulted**")
        if result["sources"]:
            for source in result["sources"]:
                st.markdown(f"- {source}")
            with st.expander("View retrieved source text"):
                for excerpt in result["source_excerpts"]:
                    st.markdown(f"**{excerpt['citation']}** - *{excerpt['section']}*")
                    st.markdown(f"> {excerpt['text']}")
                    st.markdown("---")
        else:
            st.info("No matching reference documents found.")
    with col3:
        st.markdown("**Pathway**")
        st.markdown("Use Case: **UC3 - Lab Report**")
        st.markdown(f"Abnormal values found: **{len(result['abnormal_findings'])}**")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Interpretation")
        st.markdown(result["interpretation"])

    with col_right:
        st.subheader("Urgency Assessment")
        urgency_colors = {
            "Self-Care at Home": "green",
            "See a Doctor Soon": "orange",
            "See a Doctor Today": "red",
            "Go to Emergency": "darkred"
        }
        color = urgency_colors.get(result["urgency_level"], "gray")
        st.markdown(
            f"<h2 style='color:{color}'>{result['urgency_level']}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(f"*{result['urgency_reasoning']}*")
        if result["urgency_matched_criteria"]:
            st.markdown("**Why we flagged this:**")
            for criterion in result["urgency_matched_criteria"]:
                st.warning(criterion)
        else:
            st.info("No specific red-flag criteria matched.")

    st.divider()
    st.caption(
        "Dr. Friend is a healthcare guidance assistant, not a replacement "
        "for professional medical advice. Always consult a qualified doctor."
    )

    if st.button("Start New Consultation", type="primary"):
        for key in ALL_STATE_KEYS:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    