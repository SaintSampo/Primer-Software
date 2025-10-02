# This script demonstrates how to use the Whisper-tiny.en model with ONNX Runtime
# for speech-to-text transcription. It now includes live audio recording with
# voice activity detection (VAD).

# Before running, you must install the required libraries:
# pip install onnxruntime numpy librosa soundfile
# For recording and VAD, you also need:
# pip install PyAudio webrtcvad
# For a proper tokenizer, you need:
# pip install transformers
# For PyAudio on macOS, you may need to first install PortAudio:
# brew install portaudio
# For PyAudio on Linux, you may need to first install system dependencies:
# sudo apt-get install python3-pyaudio

import onnxruntime as ort
import numpy as np
import soundfile as sf
import librosa
import pyaudio
import webrtcvad
import wave
import os
import sys
import time # Added for wait_for_prompt
from transformers import WhisperProcessor

# --- Model and Tokenizer Initialization ---

# NOTE: You must download the ONNX model files first.
# This example assumes the model and a vocabulary file are in the same directory.
# We will use separate encoder and decoder models, which is the standard approach.
ENCODER_MODEL_PATH = "lib/whisper/encoder_model.onnx"
# We're using the simpler decoder model without a past key-value cache
DECODER_MODEL_PATH = "lib/whisper/decoder_model.onnx" 

# Global variables for PyAudio and VAD objects
audio = None
vad = None

# Create the ONNX inference sessions for the encoder and decoder
try:
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    encoder_session = ort.InferenceSession(ENCODER_MODEL_PATH)
    decoder_session = ort.InferenceSession(DECODER_MODEL_PATH)
    print("ONNX models and processor loaded successfully.")
except Exception as e:
    print(f"Error loading ONNX models or processor: {e}")
    print("Please ensure the 'encoder_model.onnx' and 'decoder_model.onnx' files exist in the same directory.")
    sys.exit()

# --- Audio Recording with VAD ---

# VAD and Recording Constants
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30  # Frame duration for VAD (must be 10, 20, or 30 ms)
CHUNK = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) # Frames per buffer
FORMAT = pyaudio.paInt16
CHANNELS = 1
SILENCE_TIMEOUT_FRAMES = 32  # About 1 second of silence (1s / 30ms = ~32 frames)
MAX_RECORD_SECONDS = 30
INPUT_DEVICE_INDEX = None

# Helper to initialize global PyAudio and VAD objects
def _init_audio_components():
    """Initializes global PyAudio and VAD objects if they aren't already."""
    global audio, vad
    if audio is None:
        audio = pyaudio.PyAudio()
    if vad is None:
        # Aggressiveness mode 3: most aggressive filtering of non-speech
        vad = webrtcvad.Vad(3)


def _record_audio_chunk():
    """
    Records a single chunk of audio from the microphone with VAD.
    The recording starts when speech is detected and stops after SILENCE_TIMEOUT_FRAMES 
    of silence or MAX_RECORD_SECONDS.
    Returns the filename of the recorded WAV file or None if no speech was detected.
    """
    _init_audio_components() # Ensure PyAudio and VAD are initialized
    
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=INPUT_DEVICE_INDEX) 

    frames = []
    speaking = False
    silence_frames = 0
    total_frames = 0
    
    # Wait for initial speech
    print("  Listening for speech to start...")
    while not speaking:
        try:
            frame_data = stream.read(CHUNK, exception_on_overflow=False)
            is_speech = vad.is_speech(frame_data, SAMPLE_RATE)
            if is_speech:
                print("  Speech detected. Recording...")
                speaking = True
                frames.append(frame_data)
        except Exception:
            # Continue to wait for speech
            continue

    # Continue recording until silence or max time is hit
    while True:
        try:
            frame_data = stream.read(CHUNK, exception_on_overflow=False)
            is_speech = vad.is_speech(frame_data, SAMPLE_RATE)

            frames.append(frame_data)
            
            if not is_speech:
                silence_frames += 1
            else:
                silence_frames = 0 # Reset counter on speech

            total_frames += 1
            
            # Stop if silence timeout is reached
            if silence_frames > SILENCE_TIMEOUT_FRAMES:
                print(f"  Detected {silence_frames * FRAME_DURATION_MS / 1000}s of silence. Chunk finished.")
                break
            
            # Stop if max time is reached
            if total_frames * FRAME_DURATION_MS / 1000 >= MAX_RECORD_SECONDS:
                print("  Maximum recording time reached. Stopping chunk.")
                break
        except KeyboardInterrupt:
            raise # Let main loop handle KeyboardInterrupt
        except Exception as e:
            print(f"Error during recording: {e}")
            break

    stream.stop_stream()
    stream.close()
    
    if not frames:
        print("  No speech was recorded in this chunk.")
        return None

    # Save to a temporary file
    temp_file = f"recorded_chunk_{int(time.time())}.wav"
    wf = wave.open(temp_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"  Audio chunk saved to '{temp_file}'")
    return temp_file


def _record_and_transcribe_chunk():
    """Records an audio chunk and returns its transcription. Cleans up the file."""
    audio_path = _record_audio_chunk()
    if not audio_path:
        return None
        
    transcribed_text = transcribe_audio_internal(audio_path)
    
    # Clean up the temporary file
    os.remove(audio_path)
    
    return transcribed_text


# --- New wait_for_prompt Function ---

def wait_for_prompt(trigger_word: str):
    """
    Continuously listens for audio, transcribes it, and returns the full 
    transcription only if it contains the specified trigger word.
    
    :param trigger_word: The word that must be spoken to return the prompt (case-insensitive).
    :return: The full transcribed text as a string, or None if interrupted.
    """
    
    print(f"\n--- Awaiting Trigger Word: '{trigger_word.upper()}' ---")
    
    try:
        while True:
            # 1. Record a chunk of speech and transcribe it
            transcribed_text = _record_and_transcribe_chunk()

            if transcribed_text:
                # 2. Check if the transcription contains the trigger word
                if trigger_word.lower() in transcribed_text.lower():
                    print("\n!!! Trigger Word Detected !!!")
                    print(f"Full Prompt: {transcribed_text.strip()}")
                    # 3. Return the prompt
                    return transcribed_text.strip()
                else:
                    print(f"Prompt thrown out (no '{trigger_word}' detected). Re-listening.")
                    print("-" * 30)
            else:
                print("No clear speech detected. Re-listening.")
                print("-" * 30)
                
    except KeyboardInterrupt:
        print("\nListening stopped by user (KeyboardInterrupt).")
        return None
    finally:
        # Important: terminate PyAudio when done with all listening
        global audio
        if audio:
            audio.terminate()
            audio = None
            print("PyAudio terminated.")

def cleanup_audio():
    """Important: terminate PyAudio when done with all listening."""
    global audio
    if audio:
        audio.terminate()
        audio = None
        print("PyAudio terminated.")

# --- Audio Preprocessing Function (Simplified) ---
# NOTE: Renamed to preprocess_audio_internal to avoid confusion with the public 
# function in the original file, though the original was only used internally anyway.
def preprocess_audio_internal(audio_path):
    """
    Loads and preprocesses an audio file for the Whisper model.
    The model expects a 16kHz, single-channel Mel spectrogram.
    """
    try:
        # Load audio data and resample to 16kHz
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        print("Please ensure the audio file exists and is a supported format.")
        return None

    # Compute the log-Mel spectrogram using the processor
    mel_spec = processor.feature_extractor(
        audio, 
        sampling_rate=16000, 
        return_tensors="np"
    ).input_features
    
    # Pad the spectrogram to a fixed size (e.g., 30 seconds)
    max_len = 3000
    if mel_spec.shape[2] > max_len:
        mel_spec = mel_spec[:, :, :max_len]
    else:
        # Note: Padding with zeros is a simple approach; proper handling (like in the original paper) 
        # may involve more complex padding or chunking for longer audio.
        padding = np.zeros((mel_spec.shape[0], mel_spec.shape[1], max_len - mel_spec.shape[2]), dtype=np.float32)
        mel_spec = np.concatenate([mel_spec, padding], axis=2)
    
    return mel_spec

# --- Main Transcription Logic (Internal Version) ---
# NOTE: Renamed the original 'transcribe_audio' to 'transcribe_audio_internal'
# to make it clear this is a backend logic function.
def transcribe_audio_internal(audio_path):
    """
    Transcribes a given audio file using the ONNX model.
    Returns the transcribed text string.
    """
    print(f"  Preprocessing audio from: {audio_path}")
    mel_spec = preprocess_audio_internal(audio_path)
    if mel_spec is None:
        return ""

    # Use the encoder model to process the audio input
    encoder_input_name = encoder_session.get_inputs()[0].name
    encoder_outputs = encoder_session.run(None, {encoder_input_name: mel_spec})
    encoder_output = encoder_outputs[0]

    # Initialize decoder input with the special tokens
    decoder_input_ids = [processor.tokenizer.convert_tokens_to_ids("<|startoftext|>")]
    # Manually define forced tokens to guide the model
    forced_tokens = [
        processor.tokenizer.convert_tokens_to_ids("<|transcribe|>"),
        processor.tokenizer.convert_tokens_to_ids("<|en|>")
    ]

    # Simple greedy decoding loop
    token_ids = []
    
    print("  Starting decoding loop...")
    for i in range(100):  # Maximum number of tokens to generate
        # Force the first few tokens to guide the model
        if i < len(forced_tokens):
            next_token = forced_tokens[i]
        else:
            # Run the decoder model to predict the next token
            decoder_outputs = decoder_session.run(
                None, 
                {
                    "input_ids": np.array([decoder_input_ids], dtype=np.int64),
                    "encoder_hidden_states": encoder_output
                }
            )
            
            logits = decoder_outputs[0]
            
            # Get the next token ID with the highest probability
            next_token = np.argmax(logits[0, -1, :])
        
        # print(f"  Step {i+1}: Next token ID is {next_token}") # Verbose output removed

        # Stop if end of sequence token is predicted
        if next_token == processor.tokenizer.eos_token_id:
            print("  End of sequence token detected. Stopping decoding.")
            break
        
        # Append the new token to the input for the next step
        decoder_input_ids.append(next_token)
        
        # We need to skip the special tokens for the final transcription
        if next_token not in processor.tokenizer.all_special_ids:
            token_ids.append(next_token)

    # Decode the token IDs to text using the processor's tokenizer
    transcribed_text = processor.tokenizer.decode(
        token_ids, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    print(f"  Transcription: '{transcribed_text.strip()}'")
    return transcribed_text.strip()

# --- Example Usage ---
if __name__ == "__main__":
    # The original logic is now a test
    print("--- Running Whisper Prompt Test ---")
    try:
        # Changed to 'whisper' for a self-contained test
        prompt = wait_for_prompt("whisper") 
        if prompt:
            print("\n--- Successful Prompt Received ---")
            print(f"Your command is: {prompt}")
    finally:
        cleanup_audio() # This is the crucial line
