import streamlit as st
import openai
import whisper
import os
import tempfile
import ffmpeg
from datetime import timedelta

# Load Whisper model
model = whisper.load_model("base")  # You can try 'small', 'medium' or 'large' for better quality

# OpenAI API Key (use streamlit secrets for deployment!)
openai.api_key = st.secrets["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"]

def format_timestamp(seconds):
    return str(timedelta(seconds=int(seconds)))

def transcribe_audio(file_path):
    result = model.transcribe(file_path, verbose=True)
    return result

def summarize_text(text):
    prompt = f"Summarize the following podcast into a short description, show notes with bullet points, and a title:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response['choices'][0]['message']['content']

# Streamlit UI
st.title("🎙️ Auto Podcast Summarizer")
st.write("Upload a podcast/audio file (MP3/WAV) and get a title, summary, and timestamps!")

audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    st.info("Transcribing audio...")
    result = transcribe_audio(temp_audio_path)

    st.success("Transcription complete!")
    full_text = result["text"]
    segments = result["segments"]

    st.info("Generating summary with GPT...")
    summary = summarize_text(full_text)
    st.success("Summary generated!")

    # Display output
    st.subheader("📝 Summary & Title")
    st.write(summary)

    st.subheader("🕒 Timestamps")
    for segment in segments:
        timestamp = format_timestamp(segment['start'])
        text = segment['text']
        st.markdown(f"**{timestamp}** — {text}")

    # Optional: download buttons
    st.download_button("Download Full Transcript", full_text, file_name="transcript.txt")
