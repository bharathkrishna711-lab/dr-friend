"""
llm/conversation.py

Manages the multi-turn conversation between user and Dr. Friend.
Decides when enough information has been collected to run prediction.

TWO RESPONSIBILITIES:
1. Generate Dr. Friend's next response in the conversation
2. Decide when to stop asking and trigger prediction
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm.openai_client import call_llm

# -----------------------------------------------------------------------
# Dr. Friend's personality and conversation instructions
# This is sent to the LLM on every turn as context
# -----------------------------------------------------------------------
DR_FRIEND_SYSTEM_PROMPT = """You are Dr. Friend, a warm and empathetic AI healthcare assistant.
Your job is to collect health information conversationally before running an analysis.

YOU NEED TO COLLECT:
- Basic info: age, gender, height, weight
- Vitals: temperature, blood pressure, heart rate, SpO2
- Symptoms: detailed symptom information

HOW TO ASK QUESTIONS:
- Listen carefully to what the patient says
- Ask intelligent follow-up questions SPECIFIC to their symptoms
- Never ask the same question twice
- Vary your questions based on conversation context
- If they mention stomach issues: ask about location, frequency, food history, fluids
- If they mention chest issues: ask about breathlessness, pain type, triggers
- If they mention fever: ask about duration, pattern, chills
- If they mention head pain: ask about location, type, light sensitivity
- Always connect your questions to what they just said
- Sound like a real doctor, not a form

CONVERSATION STYLE:
- Warm but concise — maximum 3 sentences per response
- Never repeat empathetic phrases
- Each response must move the conversation forward
- Vary your sentence starters and question styles

CONVERSATION RULES:
- Ask MAXIMUM 2 questions per response — never more
- Ask the most important question first
- Keep each response under 3 sentences
- One topic at a time — don't jump between symptoms and vitals in same message


WHEN TO STOP:
Only add [READY_TO_ANALYSE] when you have ALL of these:
- Age (must be a number, not 0)
- Gender
- Height AND weight (both must be provided)
- At least one vital sign reading
- Main symptoms described in detail

If age, height or weight are missing - keep asking. Never trigger analysis without them.

NEVER diagnose. You are collecting information only.
"""

def get_dr_friend_response(conversation_history: list) -> str:
    """
    Generates Dr. Friend's next response based on conversation so far.
    
    Args:
        conversation_history: list of {"role": "user"/"assistant", "content": str}
    
    Returns:
        str: Dr. Friend's next message
        
    WHY WE PASS FULL HISTORY:
    LLMs have no memory between calls. We send the entire
    conversation every time so the LLM knows what was already asked.
    """
    # Build prompt with system instructions + full conversation history
    conversation_text = ""
    for message in conversation_history:
        role = "Patient" if message["role"] == "user" else "Dr. Friend"
        conversation_text += f"{role}: {message['content']}\n\n"
    
    prompt = f"""{DR_FRIEND_SYSTEM_PROMPT}

CONVERSATION SO FAR:
{conversation_text}

Dr. Friend's next response:"""
    
    return call_llm(prompt)


def is_ready_to_analyse(response: str) -> bool:
    """
    Checks if Dr. Friend has collected enough information.
    
    The LLM signals readiness by adding [READY_TO_ANALYSE] to its response.
    We check for this tag and trigger the prediction pipeline.
    
    WHY A TAG INSTEAD OF COUNTING TURNS:
    Counting turns is rigid - sometimes 3 turns is enough,
    sometimes 5 are needed. The LLM judges when it has enough
    information based on the conversation content.
    """
    return "[READY_TO_ANALYSE]" in response


def clean_response(response: str) -> str:
    """
    Removes the [READY_TO_ANALYSE] tag from the response
    before showing it to the user.
    User should never see this internal tag.
    """
    return response.replace("[READY_TO_ANALYSE]", "").strip()


def generate_self_care_advice(predicted_disease: str, patient_data: dict) -> list:
    """
    Generates personalized self-care advice using LLM.
    Based on specific disease and patient's actual symptoms.
    Much better than hardcoded category-based advice.
    """
    # Build symptom summary from patient data
    present_symptoms = [
        key.replace("sym_", "").replace("_", " ")
        for key, val in patient_data.items()
        if key.startswith("sym_") and val == 1
    ]
    symptoms_str = ", ".join(present_symptoms) if present_symptoms else "not specified"

    prompt = f"""You are a healthcare assistant giving self-care advice.

Patient has been assessed with: {predicted_disease}
Their symptoms include: {symptoms_str}
Their temperature: {patient_data.get('temperature_c', 37)}°C
Their SpO2: {patient_data.get('spo2_percent', 98)}%

Give 5-6 specific, practical self-care tips for this patient.
Each tip should be one clear sentence.
Be specific to their condition and symptoms — not generic advice.
Do not diagnose or prescribe medication.
Format: return only a JSON array of strings.
Example: ["Tip 1", "Tip 2", "Tip 3"]
Return only the JSON array, nothing else."""

    try:
        response = call_llm(prompt)
        # Clean and parse
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        import json
        advice_list = json.loads(cleaned)
        return advice_list
    except Exception:
        # Fallback to generic if LLM fails
        return [
            "Rest and avoid physical exertion",
            "Stay hydrated with water and clear fluids",
            "Monitor your symptoms and seek help if they worsen",
            "Eat light, easily digestible foods",
            "See a doctor if symptoms persist beyond 48 hours"
        ]

# -----------------------------------------------------------------------
# Quick test - run directly to verify conversation works
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate a conversation
    history = []
    
    # Turn 1
    history.append({
        "role": "user",
        "content": "I have had a bad cough for 5 days and chest tightness."
    })
    response = get_dr_friend_response(history)
    print("Dr. Friend:", clean_response(response))
    print("Ready to analyse:", is_ready_to_analyse(response))
    print()
    
    # Turn 2
    history.append({"role": "assistant", "content": response})
    history.append({
        "role": "user", 
        "content": "I am 42 years old, male. Temperature 38.5, BP 124/82, oxygen 95."
    })
    response = get_dr_friend_response(history)
    print("Dr. Friend:", clean_response(response))
    print("Ready to analyse:", is_ready_to_analyse(response))

    # Turn 3
    history.append({"role": "assistant", "content": response})
    history.append({
        "role": "user",
        "content": "Height 175cm, weight 78kg. I also have shortness of breath and fatigue."
    })
    response = get_dr_friend_response(history)
    print("Dr. Friend:", clean_response(response))
    print("Ready to analyse:", is_ready_to_analyse(response))