# This script demonstrates how to use the Whisper-tiny.en model with ONNX Runtime
# for speech-to-text transcription. It now includes live audio recording with
# voice activity detection (VAD).

# Before running, you must install the required libraries:
# pip install onnxruntime numpy soundfile webrtcvad

#!/usr/bin/env python3
"""
Minimal Whisper-ONNX runner that uses:
  - OpenAI Whisper (tiny.en) pre-processing: load_audio -> pad_or_trim -> log_mel_spectrogram
  - OpenAI Whisper tokenizer post-processing: decode(tokens)
  - ONNX Runtime for encoder/decoder inference

Recommended decoder: decoder_model.onnx (plain)
Optional: decoder_model_merged.onnx (with past_key_values + use_cache_branch)
"""
# Before running, you must install the required libraries:
# pip install "optimum-onnx[onnxruntime]" transformers onnx optimum-cli export onnx --model openai/whisper-tiny-en whisper_tiny_en
# cp encoder_model.onnx and decoder_model.onnx from the export output to /lib/whisper

import argparse
import onnxruntime as ort
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
import os
import sys
import wave
import time
import whisper

ENCODER_MODEL = "encoder_model.onnx"
DECODER_MODEL = "decoder_model.onnx"
MODEL_DIR_PATH = "/home/ubuntu/projects/Primer-Software/lib/whisper"
RECORD_FILE = "recording.wav"

# ---------- Audio helpers ----------
def _list_devices():
    devices = sd.query_devices()
    lines = []
    for idx, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            lines.append(f"[{idx}] {d['name']} - inputs: {d['max_input_channels']}, samplerate: {int(d['default_samplerate'])}")
    return "\n".join(lines) if lines else "No input devices found."

# --- VAD / streaming parameters to align with Whisper + VAD ---
SAMPLE_RATE = 16000           # Whisper + WebRTC VAD standard
CHANNELS = 1                  # mono
FRAME_DURATION_MS = 20        # 10, 20, or 30 ms (20 ms recommended)
CHUNK = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # samples per frame (320 for 20 ms)
VAD_AGGRESSIVENESS = 2        # 0..3
SILENCE_TIMEOUT_MS = 800
SILENCE_TIMEOUT_FRAMES = max(1, SILENCE_TIMEOUT_MS // FRAME_DURATION_MS)
MAX_RECORD_SECONDS = 12

_vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

# ---------- Whisper-ONNX runner ----------
class WhisperONNXRunner:
    """
    Supports two decoder types:
      - Plain decoder: decoder_model.onnx (input_ids + encoder_hidden_states; no caches)
      - Merged decoder: decoder_model_merged.onnx (input_ids + encoder_hidden_states + past_key_values.* + use_cache_branch)
    Uses Whisper's official pre/post for consistency with the original pipeline.
    """

    def __init__(self, providers: str = "cpu"):
        # Locate ONNX files
        enc_path = os.path.join(MODEL_DIR_PATH, ENCODER_MODEL)
        dec_path = os.path.join(MODEL_DIR_PATH, DECODER_MODEL)
        if not (os.path.isfile(enc_path) and os.path.isfile(dec_path)):
            raise FileNotFoundError(f"Missing ONNX files under {MODEL_DIR_PATH}. Expected encoder_model.onnx and decoder_model.onnx")

        # Providers
        if providers.lower() == "cuda":
            ep = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif providers.lower() == "dml":
            ep = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            ep = ["CPUExecutionProvider"]

        print(f"🧩 Loading encoder + decoder ({providers})…")
        self.encoder_sess = ort.InferenceSession(enc_path, providers=ep)
        self.decoder_sess = ort.InferenceSession(dec_path, providers=ep)
        print("ONNX models and processor loaded successfully.")

        # Tokenizer (English-only tiny.en)
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
        self.eot_id    = self.tokenizer.eot
        self.sot_seq   = list(self.tokenizer.sot_sequence_including_notimestamps)

        # Encoder I/O
        self.enc_in  = self.encoder_sess.get_inputs()[0].name   # usually 'input_features'
        self.enc_out = self.encoder_sess.get_outputs()[0].name  # usually 'encoder_hidden_states'

        # Decoder I/O
        self.dec_inputs_list = list(self.decoder_sess.get_inputs())
        self.dec_inputs = {i.name: i for i in self.dec_inputs_list}
        self.dec_in_ids = next((i.name for i in self.dec_inputs_list if "input_ids" in i.name), None)
        self.dec_in_enc = next((i.name for i in self.dec_inputs_list if "encoder_hidden_states" in i.name), None)

        # Cache-related (merged only)
        self.dec_use_branch = next((i.name for i in self.dec_inputs_list if "use_cache_branch" in i.name), None)
        self.dec_cache_pos  = next((i.name for i in self.dec_inputs_list if "cache_position" in i.name), None)
        self.past_in_names  = [i.name for i in self.dec_inputs_list if i.name.startswith("past_key_values")]
        self.is_plain = (self.dec_use_branch is None and len(self.past_in_names) == 0)

        self.dec_outs_list = list(self.decoder_sess.get_outputs())
        self.dec_outs = [o.name for o in self.dec_outs_list]
        self.dec_logits = self.dec_outs[0]
        self.present_out_names = [o.name for o in self.dec_outs_list if o.name.startswith("present.")]

        if self.dec_in_ids is None or self.dec_in_enc is None:
            raise RuntimeError(f"Decoder missing required inputs: found {list(self.dec_inputs.keys())}")

        print("🔧 Decoder type:", "PLAIN" if self.is_plain else "MERGED")

    # --- Whisper pre-processing (OpenAI reference) ---
    def _mel(self, audio_path: str) -> np.ndarray:
        # FFmpeg loader → mono/16k → pad_or_trim → log-mel spectrogram
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel_t = whisper.log_mel_spectrogram(audio)  # torch.Tensor [80, 3000]
        mel_np = mel_t.detach().float().cpu().numpy()
        return np.expand_dims(mel_np, 0)  # [1, 80, 3000]

    # --- dtype helper ---
    def _dtype_from_onnx(self, onnx_type: str):
        t = (onnx_type or "").lower()
        if "float16" in t: return np.float16
        if "float"    in t: return np.float32
        if "int64"    in t: return np.int64
        if "bool"     in t: return np.bool_
        return np.float32

    # --- merged helpers ---
    def _zeros_for_past_input(self, inp):
        dtype = self._dtype_from_onnx(inp.type)
        shape = []
        name = inp.name
        for d in inp.shape:
            if isinstance(d, int):
                shape.append(d)
            else:
                s = str(d).lower()
                if "batch" in s: shape.append(1)
                elif "past_decoder_sequence_length" in s or ("decoder" in name and "past_key_values" in name):
                    shape.append(0)   # decoder past length = 0
                elif "encoder_sequence_length" in s or ("encoder" in name and "past_key_values" in name):
                    shape.append(0)   # encoder past length = 0
                else: shape.append(1) # heads/head_dim → default 1
        return np.zeros(shape, dtype=dtype)

    def _init_past_feed(self):
        feed = {}
        for inp in self.dec_inputs_list:
            if inp.name.startswith("past_key_values"):
                feed[inp.name] = self._zeros_for_past_input(inp)
        branch_dtype = self._dtype_from_onnx(self.dec_inputs[self.dec_use_branch].type)
        feed[self.dec_use_branch] = np.asarray([0], dtype=branch_dtype)  # first step: no past
        if self.dec_cache_pos is not None:
            pos_dtype = self._dtype_from_onnx(self.dec_inputs[self.dec_cache_pos].type)
            feed[self.dec_cache_pos] = np.asarray([0], dtype=pos_dtype)
        return feed

    def _update_past_from_present(self, dec_out_list):
        next_past = {}
        for name, arr in zip(self.dec_outs, dec_out_list):
            if name.startswith("present."):
                past_name = name.replace("present.", "past_key_values.")
                next_past[past_name] = arr
        return next_past

    def transcribe(self, audio_path: str, max_tokens: int = 448, skip_special_tokens: bool = True) -> str:
        # 1) Pre: log-mel via OpenAI Whisper
        mel = self._mel(audio_path)

        # 2) Encoder → hidden states
        enc_hidden = self.encoder_sess.run(None, {self.enc_in: mel})[0]  # [1, encoder_seq_len, hidden]

        # 3) Decoder loop (greedy)
        tokens = self.sot_seq[:]

        if self.is_plain:
            # Plain decoder: simplest & robust
            while len(tokens) < max_tokens:
                input_ids = np.asarray(tokens, dtype=np.int64).reshape(1, -1)
                feed = { self.dec_in_ids: input_ids, self.dec_in_enc: enc_hidden }
                dec_out = self.decoder_sess.run(self.dec_outs, feed)
                logits = dec_out[0]
                next_id = int(np.argmax(logits[0, -1]))
                tokens.append(next_id)
                if next_id == self.eot_id:
                    break
        else:
            # Merged decoder: init with empty past + branch=0
            input_ids = np.asarray(tokens, dtype=np.int64).reshape(1, -1)
            feed = { self.dec_in_ids: input_ids, self.dec_in_enc: enc_hidden }
            feed.update(self._init_past_feed())
            dec_out = self.decoder_sess.run(self.dec_outs, feed)
            logits = dec_out[0]
            past = self._update_past_from_present(dec_out)

            # subsequent steps: branch=1 + reuse past
            branch_dtype = self._dtype_from_onnx(self.dec_inputs[self.dec_use_branch].type)
            use_past_arr = np.asarray([1], dtype=branch_dtype)

            while len(tokens) < max_tokens:
                next_id = int(np.argmax(logits[0, -1]))
                tokens.append(next_id)
                if next_id == self.eot_id:
                    break

                input_ids = np.asarray([next_id], dtype=np.int64).reshape(1, -1)
                feed = {
                    self.dec_in_ids: input_ids,
                    self.dec_in_enc: enc_hidden,
                    self.dec_use_branch: use_past_arr,
                    **past,
                }
                dec_out = self.decoder_sess.run(self.dec_outs, feed)
                logits = dec_out[0]
                past = self._update_past_from_present(dec_out)

        # 4) Post: Whisper tokenizer -> text
        return self.tokenizer.decode(tokens).strip()

def _record_audio_chunk_sd(device=None):
    """
    Records a single chunk of speech using WebRTC VAD with sounddevice.
    Starts when speech is detected and stops after trailing silence or max time.
    Returns path to a temp WAV (16-bit PCM, mono, 16k) or None if no speech.
    """
    try:
        sd.check_input_settings(device=device, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    except Exception as e:
        print("Device check failed:", e)
        print("Available input devices:\n", _list_devices())
        return None

    print("  Listening for speech to start…")
    frames_bytes = []
    speaking = False
    silence_frames = 0
    total_frames = 0
    max_frames = int(MAX_RECORD_SECONDS * 1000 / FRAME_DURATION_MS)

    with sd.RawInputStream(samplerate=SAMPLE_RATE,
                           blocksize=CHUNK,
                           channels=CHANNELS,
                           dtype='int16',
                           device=device) as stream:
        while True:
            data, overflowed = stream.read(CHUNK)  # raw bytes (int16 mono)
            is_speech = _vad.is_speech(data, SAMPLE_RATE)

            if not speaking:
                if is_speech:
                    print("  Speech detected. Recording…")
                    speaking = True
                    frames_bytes.append(data)
                continue

            frames_bytes.append(data)
            silence_frames = 0 if is_speech else (silence_frames + 1)
            total_frames += 1

            if silence_frames > SILENCE_TIMEOUT_FRAMES:
                print(f"  Detected {(silence_frames * FRAME_DURATION_MS) / 1000:.2f}s of silence. Chunk finished.")
                break
            if total_frames >= max_frames:
                print("  Maximum recording time reached. Stopping chunk.")
                break

    if not frames_bytes:
        print("  No speech was recorded in this chunk.")
        return None

    # Save the captured bytes as a WAV file compatible with your playback & Whisper
    temp_file = f"recorded_chunk_{int(time.time())}.wav"
    with wave.open(temp_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames_bytes))

    print(f"  Audio chunk saved to '{temp_file}'")

    return temp_file

def _record_and_transcribe_chunk(runner, device=None):
        """
        Record a VAD chunk → transcribe with WhisperONNXRunner → cleanup temp WAV.
        Returns transcription string or None.
        """
        audio_path = _record_audio_chunk_sd(device=device)
        if not audio_path:
            return None
        try:
            text = runner.transcribe(audio_path)
        finally:
            try:
                os.remove(audio_path)
            except Exception:
                pass
        return text

def wait_for_prompt(trigger_word: str, runner, device=None):
    """
    Loop: record chunk -> transcribe -> return full transcription once it contains trigger_word.
    Returns the transcription (prompt) string, or None on KeyboardInterrupt.
    """
    print(f"\n--- Awaiting Trigger Word: '{trigger_word.upper()}' ---")
    try:
        while True:
            prompt = _record_and_transcribe_chunk(runner, device=device)
            if prompt:
                if trigger_word.lower() in prompt.lower():
                    print("\n!!! Trigger Word Detected !!!")
                    print(f"Full Prompt: {prompt.strip()}")
                    return prompt.strip()
                else:
                    print(f"Prompt thrown out (no '{trigger_word}' detected). Re-listening.")
                    print("-" * 30)
            else:
                print("No clear speech detected. Re-listening.")
                print("-" * 30)
    except KeyboardInterrupt:
        print("\nListening stopped by user (KeyboardInterrupt).")
        return None
    