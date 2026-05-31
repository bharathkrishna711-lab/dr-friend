"""
app.py - Dr. Friend main entry point

This is the only file Streamlit runs directly.
All logic lives in core/ modules — this file handles UI and flow only.
"""

import streamlit as st
from core.model_loader import load_models
from core.predictor import predict_disease
from core.urgency_engine import assess_urgency

# -----------------------------------------------------------------------
# Page config MUST be the first Streamlit call in the script
# layout="wide" gives more horizontal space for results panel
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Dr. Friend - AI Healthcare Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------
# Load models once at startup using Streamlit's cache
# @st.cache_resource inside load_models keeps models in memory
# Without caching, models reload on every button click — very slow
# -----------------------------------------------------------------------
models = load_models()

# -----------------------------------------------------------------------
# Session state initialization
# Streamlit reruns entire script on every user interaction
# st.session_state persists data across those reruns
#
# stage controls which screen is shown:
#   "chat"      -> conversation is ongoing
#   "analysing" -> processing, show spinner
#   "results"   -> show prediction output
# -----------------------------------------------------------------------
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


# -----------------------------------------------------------------------
# Sidebar: model info panel
# Shows model is loaded and ready — useful for VIVA demonstration
# -----------------------------------------------------------------------
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

    # Reset button - clears session and starts fresh
    if st.button("Reset Conversation", use_container_width=True):
        for key in ["messages", "stage", "patient_data",
                    "prediction_result", "urgency_result"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.divider()
    st.caption("BITS Pilani WILP")
    st.caption("Bharath Krishna | 2024DA04347")

    # -----------------------------------------------------------------------
# Main content area
# -----------------------------------------------------------------------
st.title("Dr. Friend")
st.caption("AI-Powered Healthcare Guidance and Triage Assistant")
st.divider()

# -----------------------------------------------------------------------
# SCREEN ROUTER
# Renders different screens based on current stage
# This pattern keeps each screen's code separate and clean
# -----------------------------------------------------------------------
if st.session_state.stage == "chat":
    # Screen 1: Chat interface
   if st.session_state.stage == "chat":

    # Render existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Show opening message if conversation just started
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

    # Chat input box
    user_input = st.chat_input("Type your symptoms here...")

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get Dr. Friend's response from LLM
        with st.spinner("Dr. Friend is thinking..."):
            from llm.conversation import get_dr_friend_response, is_ready_to_analyse, clean_response
            from llm.entity_extractor import extract_patient_data

            response = get_dr_friend_response(st.session_state.messages)
            cleaned = clean_response(response)

        # Show Dr. Friend's response
        st.session_state.messages.append(
            {"role": "assistant", "content": cleaned}
        )
        with st.chat_message("assistant"):
            st.markdown(cleaned)

        # Check if enough info collected to run prediction
        if is_ready_to_analyse(response):
            # Build full conversation text for entity extraction
            conversation_text = ""
            for msg in st.session_state.messages:
                role = "Patient" if msg["role"] == "user" else "Dr. Friend"
                conversation_text += f"{role}: {msg['content']}\n\n"

            # Extract patient data from conversation
            with st.spinner("Extracting health information..."):
                st.session_state.patient_data = extract_patient_data(conversation_text)

            st.session_state.stage = "analysing"
            st.rerun()
        # -----------------------------------------------------------------------
    # Test trigger - Week 3 only
    # Replaced by Gemini conversation in Week 4
    # -----------------------------------------------------------------------
    st.divider()
    st.caption("Week 3 test mode - run pipeline with sample patient data")

    if st.button("Run Test Prediction", type="primary"):
        st.session_state.patient_data = {
            'age': 42,
            'gender': 'Male',
            'height_cm': 175,
            'weight_kg': 78,
            'bmi': 25.5,
            'temperature_c': 38.5,
            'heart_rate_bpm': 94,
            'systolic_bp': 124,
            'diastolic_bp': 82,
            'spo2_percent': 95.0,
            'respiratory_rate': 20,
            'blood_glucose_mgdl': 95,
            'symptom_duration_days': 5,
            'sym_cough': 1,
            'sym_fever': 1,
            'sym_shortness_of_breath': 1,
            'sym_fatigue': 1,
            'sym_chest_tightness': 1,
            'sym_productive_cough': 1,
            'sym_chills': 1,
            'sym_weakness': 1,
            'sym_wheezing': 1
        }
        st.session_state.stage = "analysing"
        st.rerun()

elif st.session_state.stage == "analysing":
    with st.spinner("Analysing your symptoms..."):
            try:
                prediction = predict_disease(
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

elif st.session_state.stage == "results":
    # Screen 3: Results display
   
    prediction = st.session_state.prediction_result
    urgency = st.session_state.urgency_result

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("What Might Be Going On")
        st.metric(
            label="Most Likely Condition",
            value=prediction["top_disease"],
            delta=f"{round(prediction['top_confidence'] * 100, 1)}% confidence"
        )
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

    st.divider()
    st.caption(
        "Dr. Friend is a healthcare guidance assistant, not a replacement "
        "for professional medical advice. Always consult a qualified doctor."
    )
    if st.button("Start New Consultation", type="primary"):
        for key in ["messages", "stage", "patient_data",
                    "prediction_result", "urgency_result"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
