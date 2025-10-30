import streamlit as st
import whisper
import tempfile
import os

st.set_page_config(page_title="🎧 Auto Caption Tool", layout="centered")

st.title("🎧 Automatic Audio Caption Tool")
st.write("Upload audio or video to generate live captions automatically — free and open-source!")

# File uploader
audio_file = st.file_uploader("Upload your audio/video file", type=["mp3", "mp4", "wav", "m4a"])

if audio_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    st.audio(audio_file)

    st.write("⏳ Transcribing... please wait (this might take 30–60 seconds)...")
    model = whisper.load_model("base")
    result = model.transcribe(tmp_path, word_timestamps=True)

    st.success("✅ Transcription complete!")
    st.write("### Transcript with timestamps:")
    for segment in result["segments"]:
        for word_info in segment["words"]:
            word = word_info["word"]
            start = word_info["start"]
            end = word_info["end"]
            st.markdown(
                f"<span style='background:#b8f5b3;padding:3px;border-radius:5px;margin:1px;'>{word}</span> "
                f"<sub>({start:.1f}s–{end:.1f}s)</sub>", unsafe_allow_html=True
            )

    # Clean up temp file
    os.remove(tmp_path)
