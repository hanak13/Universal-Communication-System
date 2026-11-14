"""
UCS — Universal Communication System (Phase 1)
"""
import os
import streamlit as st

from db import (
    init_db, save_clip, get_clips,
    distinct_labels, distinct_speakers,
)
from audio_io import build_filename, save_wav_from_file, load_wav

if "conn" not in st.session_state:
    st.session_state.conn = init_db()

conn = st.session_state.conn

st.set_page_config(page_title="UCS — Universal Communication System", layout="wide")
page = st.sidebar.radio("Navigate", ["🎙️ Record", "📚 Library"])

if page == "🎙️ Record":
    st.title("🎙️ Record a Clip")
    st.caption("Record a word or phrase. Every meaning is a *hypothesis* until verified.")

    col1, col2 = st.columns(2)
    with col1:
        label = st.text_input("Label (what you think this sound means)").strip().lower().replace(" ", "_")
    with col2:
        speaker = st.text_input("Speaker name", value="speaker1").strip().lower().replace(" ", "_")

    meaning = st.text_input("Meaning / translation (optional hypothesis)").strip()
    audio_data = st.audio_input("🎤 Press to record")

    if audio_data is not None:
        if not label:
            st.warning("Please enter a label before saving.")
        else:
            filename = build_filename(label, speaker)
            wav_bytes = audio_data.getvalue()
            save_wav_from_file(wav_bytes, filename)
            clip_id = save_clip(conn, filename=filename, label=label, speaker=speaker)
            st.success(f"✅ Saved **{filename}** (clip #{clip_id})")
            st.audio(wav_bytes, format="audio/wav")

elif page == "📚 Library":
    st.title("📚 Clip Library")
    labels = ["(all)"] + distinct_labels(conn)
    speakers = ["(all)"] + distinct_speakers(conn)

    fc1, fc2 = st.columns(2)
    with fc1:
        sel_label = st.selectbox("Filter by label", labels)
    with fc2:
        sel_speaker = st.selectbox("Filter by speaker", speakers)

    lbl = sel_label if sel_label != "(all)" else None
    spk = sel_speaker if sel_speaker != "(all)" else None
    clips = get_clips(conn, label=lbl, speaker=spk)

    if not clips:
        st.info("No clips yet. Go to **Record** to add some!")
    else:
        st.write(f"**{len(clips)}** clip(s)")
        for clip in clips:
            with st.expander(f"🔊 {clip['filename']}", expanded=False):
                wav = load_wav(clip["filename"])
                if wav:
                    st.audio(wav, format="audio/wav")
                st.markdown(
                    f"**Label:** {clip['label']}  \n"
                    f"**Speaker:** {clip['speaker']}  \n"
                    f"**Recorded:** {clip['created_at']}"
                )
