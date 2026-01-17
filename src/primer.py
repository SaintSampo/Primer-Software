#!/usr/bin/python
# -*- coding:utf-8 -*-

import time
import re
import whisper_prompt
from render_avatar import set_stage, render_paragraph
# --- IMPORT FROM OLLAMA-CHAT.PY ---
from ollama_chat import get_primer_response, INITIAL_MESSAGES_HISTORY 

# --- Configuration ---
TRIGGER_WORD = "Primer"
OLLAMA_MODEL = "tinyllama" 
RECORD_FILE = "recording.wav"

# Initialize the conversation history using the imported setup
messages_history = INITIAL_MESSAGES_HISTORY

# --- Core Logic Functions ---

def parse_and_display(ai_response: str):
    """
    Parses the mood, sets the avatar, and renders the text on the e-ink display.
    """
    moods = ["Neutral", "Laughing", "Confused", "Celebratory", "Sad", "Sleeping"]
    
    # Use regex to find the mood word at the start, case-insensitive
    match = re.match(r"(\w+): ", ai_response, re.IGNORECASE)
    
    if match and match.group(1).capitalize() in moods:
        emotion = match.group(1).capitalize()
        text = ai_response[len(match.group(0)):].strip()
    else:
        # Default to Neutral if parsing fails
        emotion = "Neutral"
        text = ai_response
    
    print(f"Parsed Emotion: {emotion}")
    print(f"Text to Display: {text}")

    # Set the avatar
    set_stage(emotion)
    time.sleep(1) # Wait for initial clear/draw to complete

    # Render the text
    render_paragraph(text)
    time.sleep(5) # Keep text on screen for a moment

def main_loop():
    """
    The main loop of the AI teacher application.
    """
    global messages_history # Need to update the global history list
    
    try:
        # Initial display state
        set_stage("Neutral")
        render_paragraph(f"Hello, I am Primer. I am listening for my name, '{TRIGGER_WORD}'.")
        time.sleep(2)
        
        while True:
            # 1. Listen for prompt
            runner = whisper_prompt.WhisperONNXRunner()
            prompt = whisper_prompt.wait_for_prompt(TRIGGER_WORD, runner, device=None)
            if TRIGGER_WORD.lower() in prompt.lower():
                print(f"\n✅ Trigger word '{TRIGGER_WORD}' detected in transcription!")
            else:
                print(f"\n❌ Trigger word '{TRIGGER_WORD}' NOT detected in transcription.")
                break

            if prompt is None: # User interrupted
                break
                
            # 2. Process the prompt
            # Trim the trigger word from the prompt before sending to LLM
            user_question = re.sub(TRIGGER_WORD, '', prompt, flags=re.IGNORECASE).strip()
            
            if not user_question:
                 ai_response = "Neutral: Yes, you called for me. What subject shall we explore?"
            else:
                print(f"\nSending user question to Ollama: {user_question}")
                 # 3. Get AI Response using the imported function
                ai_response, messages_history = get_primer_response( user_question, messages_history, OLLAMA_MODEL)

            # 4. Parse and Display
            parse_and_display(ai_response)
            
    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
    finally:
        # Clean up the display and audio resources
        set_stage("Sleeping") # Clears display and puts it to sleep

        print("Primer is signing off.")

if __name__ == "__main__":
    main_loop()

###### END OF FILE ######