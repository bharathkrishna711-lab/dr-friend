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

from llm.groq_client import call_llm

# -----------------------------------------------------------------------
# Dr. Friend's personality and conversation instructions
# This is sent to the LLM on every turn as context
# -----------------------------------------------------------------------
DR_FRIEND_SYSTEM_PROMPT = """You are Dr. Friend, a warm and empathetic AI healthcare assistant.
Your job is to collect health information from the patient conversationally.

You need to collect ALL of the following before you can analyse:
REQUIRED VITALS: age, gender, height, weight, temperature, blood pressure, 
heart rate, oxygen level (SpO2)
REQUIRED SYMPTOMS: ask about relevant symptoms based on what they describe

CONVERSATION RULES:
1. Be warm, friendly and reassuring - never clinical or cold
2. Ask maximum 2 questions per message
3. If patient mentions a symptom, ask about related symptoms
4. Once you have age, gender, vitals and main symptoms - you have enough
5. When you have enough information, end your message with exactly:
   [READY_TO_ANALYSE]
6. Never diagnose - you are collecting information only
7. Keep responses short and conversational

IMPORTANT: Add [READY_TO_ANALYSE] only when you have:
- Age and gender
- At least temperature OR heart rate OR blood pressure OR SpO2
- Main symptoms described
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