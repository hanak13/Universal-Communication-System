"""
UCS — Universal Communication System
Streamlit app (UI only). Logic lives in db.py, audio_io.py, features.py,
clustering.py, recognizer.py.
"""
import os
import json
import hashlib
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd

from db import (
    init_db, save_clip, get_clips, get_clip_by_id,
    distinct_labels, distinct_speakers,
    save_features, get_features, clips_without_features,
    save_pending_sound, get_all_pending_sounds, get_pending_sounds_with_vectors,
    next_pending_name, delete_pending_by_name, get_pending_clip_ids,
    save_word, get_word_by_id, get_all_words, get_word_by_canonical,
    update_word, confirm_word, reject_word, word_confidence,
    add_variant, get_variants, lookup_by_variant,
    log_history, undo_last_change, get_recent_history,
    save_phrase, get_phrase_by_text, get_all_phrases, update_phrase,
    save_rule, get_all_rules, get_rules_by_type, get_rule_by_id,
    update_rule, delete_rule,
    save_sequence, get_all_sequences,
    save_transcription_log,
)
from audio_io import (
    build_filename,
    save_wav_from_file,
    load_wav,
    parse_filename,
    list_recordings,
    audio_array_to_wav_bytes,
)
from config import (
    AUDIO_DIR, FEATURE_SR, HOP_LENGTH, N_FFT,
    MIN_CLIPS_FOR_CLUSTERING, DBSCAN_EPS_DEFAULT, DBSCAN_MIN_SAMPLES_DEFAULT,
    RECOGNIZE_THRESHOLD, UNKNOWN_DISTANCE_THRESHOLD, DUPLICATE_SOUND_THRESHOLD,
    LOW_EVIDENCE_THRESHOLD, WORD_TYPES,
    SPLIT_TOP_DB_DEFAULT, SPLIT_TOP_DB_MIN, SPLIT_TOP_DB_MAX,
    MIN_SEGMENT_DURATION, MAX_SEGMENTS_PER_CLIP, SEQUENCE_AUDIO_DIR_NAME,
)
from evaluation import run_evaluation, export_report, generate_report, LIMITATIONS_TEXT

# ── Initialise DB once per session ─────────────────────────────────
if "conn" not in st.session_state:
    st.session_state.conn = init_db()

conn = st.session_state.conn

# ── Sidebar navigation ────────────────────────────────────────────
st.set_page_config(page_title="UCS — Universal Communication System", layout="wide")

PAGES = [
    "🏠 Dashboard",
    "🎙️ Record",
    "📚 Library",
    "📊 Analysis",
    "🔬 Clustering",
    "❓ Pending",
    "📝 Words",
    "🔗 Sequences",
    "📏 Rules",
    "✅ Test",
    "📋 Summary",
    "📈 Evaluation",
]
page = st.sidebar.radio("Navigate", PAGES)

# ── Sidebar undo button ───────────────────────────────────────────
st.sidebar.divider()
if st.sidebar.button("↩️ Undo last change"):
    msg = undo_last_change(conn)
    if msg:
        st.sidebar.success(msg)
    else:
        st.sidebar.info("Nothing to undo.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: Dashboard (main landing page)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == "🏠 Dashboard":
    st.title("🏠 UCS — Universal Communication System")
    st.caption(
        "Record or upload a clip to get instant recognition. "
        "Every meaning is a *hypothesis* until verified."
    )

    # ── Quick stats row ────────────────────────────────────────────
    all_words = get_all_words(conn)
    all_phrases_dash = get_all_phrases(conn)
    pending_dash = get_all_pending_sounds(conn)
    all_clips_dash = get_clips(conn)

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("📦 Clips", len(all_clips_dash))
    sc2.metric("📝 Words", len(all_words))
    sc3.metric("📖 Phrases", len(all_phrases_dash))
    sc4.metric("❓ Pending", len(set(p["name"] for p in pending_dash)))

    # Cold-start hint (Task 8)
    if len(all_words) == 0:
        st.info("💡 **Cold-start hint:** Say one word at a time to teach the system.")

    st.divider()

    # ── Record or upload ───────────────────────────────────────────
    st.subheader("🎤 Quick Recognize")
    dash_mode = st.radio("Input", ["Record", "Upload WAV"], horizontal=True, key="dash_mode")

    dash_audio = None
    if dash_mode == "Record":
        dash_audio = st.audio_input("🎤 Press to record", key="dash_rec")
    else:
        dash_upload = st.file_uploader("Upload a WAV file", type=["wav"], key="dash_upload")
        if dash_upload:
            dash_audio = dash_upload

    if dash_audio is not None:
        audio_bytes = dash_audio.getvalue() if hasattr(dash_audio, "getvalue") else dash_audio.read()
        clip_hash = hashlib.md5(audio_bytes).hexdigest()[:12]

        # Reset skipped segments when clip audio changes
        if st.session_state.get("dash_active_clip_hash") != clip_hash:
            st.session_state.dash_active_clip_hash = clip_hash
            st.session_state.dash_skipped_indices = set()

        if "dash_skipped_indices" not in st.session_state:
            st.session_state.dash_skipped_indices = set()

        st.audio(audio_bytes, format="audio/wav")

        from recognizer import Recognizer
        from sequences import split_audio, extract_segment_features
        from transcribe import transcribe_file, transcribe_segment, is_available as whisper_ok
        import librosa

        # Save temp file
        seq_dir = os.path.join(os.path.dirname(AUDIO_DIR), SEQUENCE_AUDIO_DIR_NAME)
        os.makedirs(seq_dir, exist_ok=True)
        tmp_path = os.path.join(seq_dir, f"dash_{clip_hash}.wav")
        if not os.path.exists(tmp_path):
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)

        # ── "You said:" whole-clip transcription ──────────────────
        whole_transcript = None
        _whisper_live = whisper_ok()
        if _whisper_live:
            try:
                whole_transcript = transcribe_file(tmp_path)
            except Exception:
                whole_transcript = None

        if whole_transcript:
            st.markdown(f"#### 🗣️ You said: *\"{whole_transcript}\"*")
        elif _whisper_live:
            st.caption("🗣️ *(Could not transcribe audio)*")
        else:
            st.caption(
                "🗣️ *Transcription unavailable — install `faster-whisper` "
                "in the venv for automatic heard-spelling.*"
            )

        # Build recognizer from known clips
        all_c = get_clips(conn)
        clip_data_dash = []
        for c in all_c:
            feat = get_features(conn, c["id"])
            if feat:
                clip_data_dash.append({
                    "id": c["id"], "filename": c["filename"],
                    "label": c["label"], "vector": feat["vector"],
                })

        recog = Recognizer()
        if len(clip_data_dash) >= 1:
            recog.fit(
                [d["vector"] for d in clip_data_dash],
                [d["label"] for d in clip_data_dash],
            )

        # Build word lookup (canonical and variants)
        word_lookup_dash: dict[str, dict] = {}
        for w in all_words:
            word_lookup_dash[w["canonical"]] = w
            for v in get_variants(conn, w["id"]):
                word_lookup_dash[v] = w

        # Split into word segments
        try:
            segments = split_audio(tmp_path, top_db=SPLIT_TOP_DB_DEFAULT)
        except Exception:
            segments = []

        if not segments:
            y, sr = librosa.load(tmp_path, sr=FEATURE_SR, mono=True)
            if len(y) > 0:
                segments = [{
                    "start_sample": 0, "end_sample": len(y),
                    "start_sec": 0.0, "end_sec": float(len(y) / sr),
                    "y": y,
                }]

        segments = segments[:MAX_SEGMENTS_PER_CLIP]

        if not segments:
            st.warning("No audio detected in the clip.")
        else:
            # Analyze each segment left to right, transcribe unknowns
            segment_info = []
            first_unresolved_idx = None

            for idx, seg in enumerate(segments):
                vec, raw = extract_segment_features(seg["y"])
                rec = recog.recognize(vec, threshold=UNKNOWN_DISTANCE_THRESHOLD)
                matched_word = word_lookup_dash.get(rec["label"]) if not rec["is_unknown"] else None
                is_known = matched_word is not None
                is_skipped = idx in st.session_state.dash_skipped_indices
                seg_wav = audio_array_to_wav_bytes(seg["y"], sr=FEATURE_SR)

                # Transcribe unknown segments per word
                heard_guess = None
                if not is_known and not is_skipped:
                    if _whisper_live:
                        try:
                            heard_guess = transcribe_segment(seg["y"])
                        except Exception:
                            heard_guess = None

                info = {
                    "idx": idx,
                    "seg": seg,
                    "vec": vec,
                    "raw": raw,
                    "rec": rec,
                    "matched_word": matched_word,
                    "is_known": is_known,
                    "is_skipped": is_skipped,
                    "wav_bytes": seg_wav,
                    "heard_guess": heard_guess,
                }
                segment_info.append(info)

                if not is_known and not is_skipped and first_unresolved_idx is None:
                    first_unresolved_idx = idx

            # ── Word chips ─────────────────────────────────────────
            st.divider()
            st.subheader("🧩 Word Segments")
            st.caption("Known words vs heard guesses. Meanings are hypotheses.")

            chip_parts = []
            for info in segment_info:
                if info["is_known"]:
                    lbl = info["matched_word"]["canonical"]
                    chip_parts.append(
                        f'<span style="display:inline-block;margin:2px 4px;padding:4px 12px;'
                        f'border-radius:16px;background:#c8e6c9;color:#1b5e20;'
                        f'font-weight:600;">{lbl}</span>'
                    )
                elif info["is_skipped"]:
                    lbl = info.get("heard_guess") or info["rec"].get("label", "?")
                    chip_parts.append(
                        f'<span style="display:inline-block;margin:2px 4px;padding:4px 12px;'
                        f'border-radius:16px;background:#e0e0e0;color:#616161;'
                        f'font-style:italic;">{lbl} (skipped)</span>'
                    )
                else:
                    lbl = info.get("heard_guess") or f"new_word_{info['idx']+1}"
                    chip_parts.append(
                        f'<span style="display:inline-block;margin:2px 4px;padding:4px 12px;'
                        f'border-radius:16px;background:#fff9c4;color:#f57f17;'
                        f'font-weight:600;">{lbl} ?</span>'
                    )
            st.markdown("".join(chip_parts), unsafe_allow_html=True)

            # ── Segment detail rows ───────────────────────────────
            max_show_idx = first_unresolved_idx if first_unresolved_idx is not None else len(segment_info) - 1

            for i in range(max_show_idx + 1):
                info = segment_info[i]
                c_idx, c_aud, c_wd, c_mn, c_cf = st.columns([1, 3, 2, 3, 2])
                with c_idx:
                    st.markdown(f"**#{i + 1}**")
                with c_aud:
                    st.audio(info["wav_bytes"], format="audio/wav")
                with c_wd:
                    if info["is_known"]:
                        st.markdown(f"**`{info['matched_word']['canonical']}`**")
                    elif info["is_skipped"]:
                        guess = info.get("heard_guess") or info["rec"].get("label", "unknown")
                        st.markdown(f"*Skipped* (`{guess}`)")
                    else:
                        guess = info.get("heard_guess")
                        if guess:
                            st.markdown(f"🎧 heard: **`{guess}`**")
                        else:
                            st.markdown("❓ **Unknown**")
                with c_mn:
                    if info["is_known"]:
                        st.write(f"*{info['matched_word'].get('meaning') or '?'}*")
                    elif info["is_skipped"]:
                        st.caption("Pending")
                    else:
                        st.caption("Needs meaning")
                with c_cf:
                    if info["is_known"]:
                        conf_val, conf_tot = word_confidence(info["matched_word"])
                        status = info["matched_word"].get("status", "unverified")
                        badge = "⚠️ Unverified" if status == "unverified" else "✅ Verified"
                        if conf_tot < LOW_EVIDENCE_THRESHOLD:
                            st.caption(f"{badge} (low evidence)")
                        else:
                            st.caption(f"{badge} ({conf_val:.0%})")
                    else:
                        st.caption("—")

            # Inline form directly under the first unresolved segment
            if first_unresolved_idx is not None:
                target_info = segment_info[first_unresolved_idx]
                heard = target_info.get("heard_guess") or ""
                st.markdown("---")
                st.subheader(f"✏️ Teach Meaning for Segment #{first_unresolved_idx + 1}")

                # Show editable heard-as text box for unknown segments
                if heard:
                    edited_heard = st.text_input(
                        "🎧 Heard as (edit if wrong)",
                        value=heard,
                        key=f"heard_edit_{first_unresolved_idx}",
                    ).strip().lower()
                else:
                    edited_heard = ""
                    if not _whisper_live:
                        st.caption(
                            "⚠️ Transcription engine not installed. "
                            "Install `faster-whisper` for automatic heard-spelling."
                        )

                # Default Spelling from heard/edited text
                default_spelling = edited_heard if edited_heard else ""

                # Check duplicate candidate
                close_cand = None
                close_dist = None
                if target_info["rec"]["top3"]:
                    top_lbl, top_dst = target_info["rec"]["top3"][0]
                    if top_dst <= DUPLICATE_SOUND_THRESHOLD:
                        close_cand = top_lbl
                        close_dist = top_dst

                if close_cand:
                    st.info(f"💡 This sounds like **'{close_cand}'** (distance: {close_dist:.1f}). Same word?")

                with st.container():
                    f_col1, f_col2 = st.columns(2)
                    with f_col1:
                        spelling_in = st.text_input(
                            "Spelling",
                            value=default_spelling,
                            key=f"inline_spell_{first_unresolved_idx}",
                            placeholder="e.g. inu",
                        ).strip().lower()
                    with f_col2:
                        meaning_in = st.text_input(
                            "English meaning (hypothesis)",
                            key=f"inline_mean_{first_unresolved_idx}",
                            placeholder="e.g. you",
                        ).strip()

                    f_col3, f_col4 = st.columns(2)
                    with f_col3:
                        type_in = st.selectbox(
                            "Word type",
                            WORD_TYPES,
                            key=f"inline_type_{first_unresolved_idx}",
                        )
                    with f_col4:
                        pn_in = st.text_input(
                            "Person / Number (optional)",
                            key=f"inline_pn_{first_unresolved_idx}",
                            placeholder="e.g. 2s",
                        ).strip()

                    b1, b2, b3, b4 = st.columns([2, 2, 2, 3])
                    with b1:
                        if st.button("💾 Save word", key=f"btn_save_{first_unresolved_idx}", type="primary"):
                            if not spelling_in:
                                st.error("Please provide a spelling for this word.")
                            else:
                                canon = spelling_in.replace(" ", "_")
                                # Log transcription guess vs user edit
                                save_transcription_log(
                                    conn,
                                    segment_index=first_unresolved_idx,
                                    heard_guess=heard or None,
                                    user_edit=edited_heard if edited_heard != heard else None,
                                    final_spelling=canon,
                                    clip_hash=clip_hash,
                                )
                                # Save or update word
                                existing_w = lookup_by_variant(conn, canon)
                                if existing_w is None:
                                    save_word(
                                        conn,
                                        canonical=canon,
                                        meaning=meaning_in or None,
                                        word_type=type_in,
                                        person_number=pn_in or None,
                                        status="unverified",
                                    )
                                else:
                                    if meaning_in and not existing_w.get("meaning"):
                                        update_word(conn, existing_w["id"], meaning=meaning_in)

                                # Save segment audio as clip
                                speaker = "speaker1"
                                filename = build_filename(canon, speaker)
                                save_wav_from_file(target_info["wav_bytes"], filename)
                                clip_id = save_clip(conn, filename=filename, label=canon, speaker=speaker)
                                save_features(conn, clip_id, target_info["vec"], target_info["raw"]["duration"])

                                # Link pending sound if matched
                                if target_info["rec"].get("pending_name"):
                                    delete_pending_by_name(conn, target_info["rec"]["pending_name"])

                                st.success(f"✅ Saved word **{canon}** (hypothesis: *{meaning_in or '?'}*)!")
                                st.rerun()

                    with b2:
                        if st.button("⏭️ Skip for now", key=f"btn_skip_{first_unresolved_idx}"):
                            pend_name = next_pending_name(conn)
                            save_pending_sound(conn, name=pend_name, vector=target_info["vec"])
                            st.session_state.dash_skipped_indices.add(first_unresolved_idx)
                            # Log skip
                            save_transcription_log(
                                conn,
                                segment_index=first_unresolved_idx,
                                heard_guess=heard or None,
                                user_edit=None,
                                final_spelling=pend_name,
                                clip_hash=clip_hash,
                            )
                            st.info(f"Skipped for now (saved as **{pend_name}** in Pending).")
                            st.rerun()

                    with b3:
                        if all_words:
                            exist_words = [w["canonical"] for w in all_words]
                            selected_w = st.selectbox(
                                "Known word",
                                exist_words,
                                key=f"sel_same_{first_unresolved_idx}",
                                label_visibility="collapsed",
                            )
                        else:
                            selected_w = None

                    with b4:
                        if selected_w and st.button("🔗 Same as existing word", key=f"btn_same_{first_unresolved_idx}"):
                            speaker = "speaker1"
                            filename = build_filename(selected_w, speaker)
                            save_wav_from_file(target_info["wav_bytes"], filename)
                            clip_id = save_clip(conn, filename=filename, label=selected_w, speaker=speaker)
                            save_features(conn, clip_id, target_info["vec"], target_info["raw"]["duration"])
                            # Log
                            save_transcription_log(
                                conn,
                                segment_index=first_unresolved_idx,
                                heard_guess=heard or None,
                                user_edit=None,
                                final_spelling=selected_w,
                                clip_hash=clip_hash,
                            )
                            st.success(f"✅ Added audio example to existing word **{selected_w}**!")
                            st.rerun()

            else:
                # All segments are resolved!
                st.markdown("---")
                st.subheader("🎉 Sequence Recognition Complete")

                resolved_labels = []
                word_meanings = []
                for info in segment_info:
                    if info["is_known"]:
                        canon = info["matched_word"]["canonical"]
                        resolved_labels.append(canon)
                        word_meanings.append(info["matched_word"].get("meaning") or "?")
                    else:
                        lbl = info.get("heard_guess") or info["rec"].get("label", "new_word")
                        resolved_labels.append(lbl)
                        word_meanings.append("[?]")

                full_text = " ".join(resolved_labels)
                st.markdown(f"### 🗣️ Recognized: `{full_text}`")

                joined_meanings = " ".join(word_meanings)
                st.markdown(f"**Word-by-word English hypothesis:** *{joined_meanings}*")

                saved_phrase = get_phrase_by_text(conn, full_text)
                if saved_phrase:
                    st.success(f"📖 **Phrase hypothesis:** *{saved_phrase.get('meaning') or '?'}*")
                else:
                    st.warning("🆕 **New sentence found** — no phrase meaning saved yet.")
                    from rule_engine import arrange_sentence
                    all_rules = get_all_rules(conn)
                    arranged = arrange_sentence(resolved_labels, word_lookup_dash, all_rules)
                    if arranged.get("ranked_suggestions"):
                        st.markdown("**Rule-based suggestions (tentative):**")
                        for sug in arranged["ranked_suggestions"][:3]:
                            st.write(f"  → *{sug['suggestion']}* — {sug['reason']}")

                # Sentence Right / Wrong buttons
                st.markdown("#### Sentence Feedback")
                col_r, col_w = st.columns(2)
                with col_r:
                    if st.button("👍 Right", key="dash_sent_right_btn", type="primary"):
                        if saved_phrase:
                            update_phrase(conn, saved_phrase["id"], status="verified")
                        else:
                            best_meaning = joined_meanings
                            from rule_engine import arrange_sentence
                            all_rules = get_all_rules(conn)
                            arr = arrange_sentence(resolved_labels, word_lookup_dash, all_rules)
                            if arr.get("ranked_suggestions"):
                                best_meaning = arr["ranked_suggestions"][0]["suggestion"]
                            save_phrase(conn, text=full_text, meaning=best_meaning, status="verified")
                        st.success("✅ Phrase confirmed and verified!")
                        st.rerun()

                with col_w:
                    if st.button("👎 Wrong", key="dash_sent_wrong_btn"):
                        st.session_state["dash_phrase_wrong"] = True

                if st.session_state.get("dash_phrase_wrong"):
                    corr_phrase = st.text_input("Correct phrase meaning?", key="dash_phrase_corr_inp")
                    if st.button("Submit Phrase Correction", key="dash_phrase_corr_submit"):
                        if saved_phrase:
                            update_phrase(conn, saved_phrase["id"], meaning=corr_phrase, status="verified")
                        else:
                            save_phrase(conn, text=full_text, meaning=corr_phrase, status="verified")
                        st.session_state["dash_phrase_wrong"] = False
                        st.success("Saved phrase correction!")
                        st.rerun()

                # Whole-clip meaning input
                st.divider()
                st.markdown("#### 💡 Save Whole-Clip Meaning")
                c_wc_inp, c_wc_btn = st.columns([3, 1])
                with c_wc_inp:
                    whole_meaning = st.text_input(
                        "Whole-clip meaning (English)",
                        placeholder="e.g. How are you?",
                        key="dash_whole_meaning_inp",
                    ).strip()
                with c_wc_btn:
                    st.write("")
                    if st.button("Save as phrase", key="dash_save_as_phrase_btn"):
                        if whole_meaning:
                            if saved_phrase:
                                update_phrase(conn, saved_phrase["id"], meaning=whole_meaning, status="verified")
                            else:
                                save_phrase(conn, text=full_text, meaning=whole_meaning, status="verified")
                            st.success(f"✅ Saved phrase: **{full_text}** = *{whole_meaning}*")
                            st.rerun()
                        else:
                            st.error("Please enter a meaning first.")

    # ── Pending sounds list ────────────────────────────────────────
    if pending_dash:
        st.divider()
        st.subheader("❓ Pending Unknown Sounds")
        groups = {}
        for p in pending_dash:
            groups.setdefault(p["name"], []).append(p)
        for name, items in list(groups.items())[:10]:
            st.caption(f"🔵 **{name}** — {len(items)} clip(s)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: Record
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🎙️ Record":
    st.title("🎙️ Record a Clip")
    st.caption(
        "Record a word or phrase. Every meaning is a *hypothesis* until verified."
    )

    col1, col2 = st.columns(2)
    with col1:
        label = st.text_input(
            "Label (what you think this sound means)",
            placeholder="e.g. hello, thank_you",
        ).strip().lower().replace(" ", "_")
    with col2:
        speaker = st.text_input(
            "Speaker name",
            value="speaker1",
        ).strip().lower().replace(" ", "_")

    meaning = st.text_input(
        "Meaning / translation (optional hypothesis)",
        placeholder="e.g. 'a greeting used in the morning'",
    ).strip()

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: Library
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
                else:
                    st.warning("Audio file missing on disk.")
                st.markdown(
                    f"**Label:** {clip['label']}  \n"
                    f"**Speaker:** {clip['speaker']}  \n"
                    f"**Mic:** {clip['mic_type']}  \n"
                    f"**Recorded:** {clip['created_at']}"
                )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📊 Analysis":
    st.title("📊 Clip Analysis")

    pending = clips_without_features(conn)
    if pending:
        st.info(f"**{len(pending)}** clip(s) have no features yet.")
        if st.button("⚡ Analyze all clips"):
            from features import extract_features
            progress = st.progress(0, text="Analyzing…")
            for i, clip in enumerate(pending):
                clip_path = os.path.join(AUDIO_DIR, clip["filename"])
                if os.path.isfile(clip_path):
                    try:
                        vector, raw = extract_features(clip_path)
                        save_features(conn, clip["id"], vector, raw["duration"])
                    except Exception as e:
                        st.warning(f"⚠️ {clip['filename']}: {e}")
                else:
                    st.warning(f"⚠️ File missing: {clip['filename']}")
                progress.progress((i + 1) / len(pending),
                                  text=f"Analyzed {i+1}/{len(pending)}")
            st.success("✅ All clips analyzed!")
            st.rerun()
    else:
        all_clips = get_clips(conn)
        if all_clips:
            st.success("All clips have features.")
        else:
            st.info("No clips yet. Go to **Record** to add some!")

    st.divider()

    all_clips = get_clips(conn)
    if not all_clips:
        st.stop()

    clip_names = {c["filename"]: c["id"] for c in all_clips}
    selected_file = st.selectbox("Select clip", list(clip_names.keys()))

    if selected_file:
        clip_id = clip_names[selected_file]
        clip_path = os.path.join(AUDIO_DIR, selected_file)

        wav = load_wav(selected_file)
        if wav:
            st.audio(wav, format="audio/wav")

        feat = get_features(conn, clip_id)
        if feat is None:
            st.warning("No features yet. Click **Analyze all clips** above.")
            st.stop()

        @st.cache_data(show_spinner="Generating plots…")
        def _compute_plots(path: str):
            from features import extract_features as _ef
            _, raw = _ef(path)
            return raw

        if not os.path.isfile(clip_path):
            st.error("Audio file not found on disk.")
            st.stop()

        raw = _compute_plots(clip_path)

        import matplotlib.pyplot as plt
        import librosa.display

        st.subheader("Waveform")
        fig_wave, ax_wave = plt.subplots(figsize=(10, 2.5))
        librosa.display.waveshow(raw["y"], sr=raw["sr"], ax=ax_wave)
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Amplitude")
        st.pyplot(fig_wave)
        plt.close(fig_wave)

        st.subheader("Spectrogram")
        fig_spec, ax_spec = plt.subplots(figsize=(10, 3.5))
        S_db = librosa.amplitude_to_db(raw["stft"], ref=np.max)
        librosa.display.specshow(
            S_db, sr=raw["sr"], hop_length=HOP_LENGTH,
            x_axis="time", y_axis="hz", ax=ax_spec,
        )
        ax_spec.set_title("Spectrogram (dB)")
        fig_spec.colorbar(ax_spec.images[0], ax=ax_spec, format="%+2.0f dB")
        st.pyplot(fig_spec)
        plt.close(fig_spec)

        st.subheader("MFCC Heatmap")
        fig_mfcc, ax_mfcc = plt.subplots(figsize=(10, 3.5))
        librosa.display.specshow(
            raw["mfcc_matrix"], sr=raw["sr"], hop_length=HOP_LENGTH,
            x_axis="time", ax=ax_mfcc,
        )
        ax_mfcc.set_title("MFCC")
        ax_mfcc.set_ylabel("Coefficient")
        fig_mfcc.colorbar(ax_mfcc.images[0], ax=ax_mfcc)
        st.pyplot(fig_mfcc)
        plt.close(fig_mfcc)

        st.subheader("Feature Values")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Duration", f"{raw['duration']:.3f} s")
            st.metric("Pitch Mean", f"{raw['pitch_mean']:.1f} Hz")
            st.metric("Pitch Std", f"{raw['pitch_std']:.1f} Hz")
        with col_b:
            st.metric("Centroid Mean", f"{raw['centroid_mean']:.1f} Hz")
            st.metric("Centroid Std", f"{raw['centroid_std']:.1f} Hz")
        with col_c:
            st.metric("Vector Length", len(feat["vector"]))

        with st.expander("Full feature vector"):
            st.json(feat["vector"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: Clustering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🔬 Clustering":
    st.title("🔬 Clustering & Recognition")

    all_clips = get_clips(conn)
    clip_data = []
    for c in all_clips:
        feat = get_features(conn, c["id"])
        if feat:
            clip_data.append({
                "id": c["id"], "filename": c["filename"],
                "label": c["label"], "vector": feat["vector"],
            })

    if len(clip_data) < MIN_CLIPS_FOR_CLUSTERING:
        st.warning(
            f"Need at least **{MIN_CLIPS_FOR_CLUSTERING}** clips with features. "
            f"Currently have **{len(clip_data)}**."
        )
        st.stop()

    labels_list = [d["label"] for d in clip_data]
    filenames_list = [d["filename"] for d in clip_data]
    vectors = np.array([d["vector"] for d in clip_data], dtype=np.float64)
    unique_labels = sorted(set(labels_list))
    n_labels = len(unique_labels)

    if n_labels < 2:
        st.warning("Need at least **2 different labels** for clustering.")
        st.stop()

    from clustering import (
        scale_vectors, run_kmeans, run_dbscan,
        reduce_pca, reduce_tsne,
        clustering_metrics, cluster_vs_label_table, find_outliers,
        fit_scaler,
    )
    from recognizer import Recognizer, find_or_create_pending_name
    import matplotlib.pyplot as plt

    scaler = fit_scaler(vectors)
    X_scaled = scale_vectors(vectors, scaler)

    st.subheader("Clustering Parameters")
    algo_col, param_col = st.columns(2)
    with algo_col:
        algo = st.radio("Algorithm", ["KMeans", "DBSCAN"])
    with param_col:
        if algo == "KMeans":
            k = st.slider("k (clusters)", 2, max(10, n_labels * 2), n_labels)
            cluster_labels = run_kmeans(X_scaled, k)
        else:
            eps = st.slider("eps", 0.5, 20.0, DBSCAN_EPS_DEFAULT, 0.5)
            min_samp = st.slider("min_samples", 1, 10, DBSCAN_MIN_SAMPLES_DEFAULT)
            cluster_labels = run_dbscan(X_scaled, eps, min_samp)

    st.subheader("Metrics")
    n_clusters_found = len(set(cluster_labels))
    if n_clusters_found < 2:
        st.warning(f"Only **{n_clusters_found}** cluster(s) found — adjust parameters.")
    else:
        metrics = clustering_metrics(labels_list, cluster_labels)
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Adjusted Rand Index", f"{metrics['ari']:.3f}")
        with mc2:
            st.metric("Homogeneity", f"{metrics['homogeneity']:.3f}")

    st.subheader("Cluster × Label")
    cross = cluster_vs_label_table(labels_list, cluster_labels)
    all_cluster_ids = sorted(set(int(c) for c in cluster_labels))
    rows = []
    for lbl in unique_labels:
        row = {"Label": lbl}
        for cid in all_cluster_ids:
            row[f"Cluster {cid}"] = cross.get(lbl, {}).get(cid, 0)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("2D Projections")
    pca_2d = reduce_pca(X_scaled, 2)
    if pca_2d.shape[1] < 2:
        pca_2d = np.column_stack([pca_2d, np.zeros(pca_2d.shape[0])])

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.caption("PCA — coloured by cluster")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.scatter(pca_2d[:, 0], pca_2d[:, 1], c=cluster_labels, cmap="tab10", alpha=0.7)
        ax1.set_title("PCA (cluster)")
        st.pyplot(fig1)
        plt.close(fig1)
    with pcol2:
        st.caption("PCA — coloured by label")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        label_ints = [unique_labels.index(l) for l in labels_list]
        ax2.scatter(pca_2d[:, 0], pca_2d[:, 1], c=label_ints, cmap="tab10", alpha=0.7)
        ax2.set_title("PCA (label)")
        st.pyplot(fig2)
        plt.close(fig2)

    if len(clip_data) >= 3:
        tsne_2d = reduce_tsne(X_scaled, 2)
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.caption("t-SNE — coloured by cluster")
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            ax3.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=cluster_labels, cmap="tab10", alpha=0.7)
            st.pyplot(fig3)
            plt.close(fig3)
        with tcol2:
            st.caption("t-SNE — coloured by label")
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            ax4.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=label_ints, cmap="tab10", alpha=0.7)
            st.pyplot(fig4)
            plt.close(fig4)

    outliers = find_outliers(labels_list, cluster_labels, filenames_list)
    if outliers:
        st.subheader("⚠️ Outlier Clips")
        st.dataframe(pd.DataFrame(outliers), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🎯 Test Recognition")
    recognizer = Recognizer()
    recognizer.fit([d["vector"] for d in clip_data], labels_list)

    if recognizer.is_fitted:
        test_clip_name = st.selectbox("Test clip", filenames_list, key="recog_test")
        if test_clip_name:
            idx = filenames_list.index(test_clip_name)
            result = recognizer.recognize(clip_data[idx]["vector"])
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.metric("Predicted", result["label"])
            with rc2:
                st.metric("Distance", f"{result['distance']:.2f}")
            with rc3:
                st.metric("Actual", labels_list[idx])
            if result["top3"]:
                st.write("**Top-3:**", ", ".join(
                    f"{lbl} ({d:.2f})" for lbl, d in result["top3"]
                ))
            if result["is_unknown"]:
                st.info("Distance exceeds threshold — would be marked **unknown**.")
                if st.button("Save as pending sound"):
                    pending_existing = get_pending_sounds_with_vectors(conn)
                    name, reused = find_or_create_pending_name(
                        clip_data[idx]["vector"], pending_existing,
                    )
                    save_pending_sound(conn, name,
                                       clip_id=clip_data[idx]["id"],
                                       vector=clip_data[idx]["vector"])
                    if reused:
                        st.success(f"Matched existing pending group: **{name}**")
                    else:
                        st.success(f"Created new pending sound: **{name}**")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: Pending  (with "Name this sound" → creates a Word)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "❓ Pending":
    st.title("❓ Pending / Unknown Sounds")
    st.caption(
        "Sounds the recognizer couldn't match. "
        "Name one to create a word hypothesis."
    )

    pending = get_all_pending_sounds(conn)
    if not pending:
        st.info("No pending sounds yet.")
        st.stop()

    groups: dict[str, list] = {}
    for p in pending:
        groups.setdefault(p["name"], []).append(p)

    for name, items in groups.items():
        with st.expander(f"🔵 {name} ({len(items)} clip(s))", expanded=False):
            for item in items:
                clip_id = item.get("clip_id")
                if clip_id:
                    clip = get_clip_by_id(conn, clip_id)
                    if clip:
                        wav = load_wav(clip["filename"])
                        if wav:
                            st.audio(wav, format="audio/wav")
                        st.caption(f"{clip['filename']} — label: {clip['label']}")

            # ── Name this pending sound → create a Word ────────────
            st.markdown("---")
            st.markdown("**🏷️ Name this sound (create a word hypothesis)**")

            canonical = st.text_input(
                "Spelling (canonical)", key=f"canon_{name}",
                placeholder="e.g. inu",
            ).strip().lower()

            variants_raw = st.text_input(
                "Other spellings (comma-separated)", key=f"var_{name}",
                placeholder="e.g. eenu, inuu",
            ).strip()

            meaning = st.text_input(
                "Meaning hypothesis", key=f"mean_{name}",
                placeholder="e.g. dog",
            ).strip()

            wtype = st.selectbox(
                "Word type", WORD_TYPES, key=f"wtype_{name}",
            )

            pn = st.text_input(
                "Person / number (optional)", key=f"pn_{name}",
                placeholder="e.g. 3rd singular",
            ).strip()

            if st.button(f"✅ Create word from {name}", key=f"create_{name}"):
                if not canonical:
                    st.warning("Enter a spelling first.")
                else:
                    # Check for duplicates
                    existing = get_word_by_canonical(conn, canonical)
                    if existing:
                        st.error(f"A word with spelling **{canonical}** already exists.")
                    else:
                        word_id = save_word(
                            conn, canonical=canonical,
                            meaning=meaning or None,
                            word_type=wtype,
                            person_number=pn or None,
                            status="unverified",
                        )
                        log_history(conn, "words", word_id, None,
                                    json.dumps({"action": "created_from_pending",
                                                "pending_name": name}))

                        # Add variants
                        if variants_raw:
                            for v in variants_raw.split(","):
                                v = v.strip().lower()
                                if v and v != canonical:
                                    add_variant(conn, word_id, v)

                        # Link clips: update their label to the canonical spelling
                        clip_ids = get_pending_clip_ids(conn, name)
                        for cid in clip_ids:
                            conn.execute(
                                "UPDATE clips SET label = ? WHERE id = ?",
                                (canonical, cid),
                            )
                        conn.commit()

                        # Remove from pending
                        delete_pending_by_name(conn, name)

                        # Retrain recognizer with newly linked clips
                        all_c = get_clips(conn)
                        recog_vecs, recog_lbls = [], []
                        for c in all_c:
                            feat = get_features(conn, c["id"])
                            if feat:
                                recog_vecs.append(feat["vector"])
                                recog_lbls.append(c["label"])
                        if len(recog_vecs) >= 2:
                            from recognizer import Recognizer
                            recog = Recognizer()
                            recog.fit(recog_vecs, recog_lbls)

                        st.success(
                            f"✅ Created word **{canonical}** "
                            f"(hypothesis: *{meaning or '?'}*) "
                            f"from {name} and retrained recognizer."
                        )
                        st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: Words
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📝 Words":
    st.title("📝 Word Dictionary")
    st.caption("All hypothesised words. Edit meanings, add variants, change status.")

    words = get_all_words(conn)
    if not words:
        st.info("No words yet. Name a pending sound to create one.")
        st.stop()

    for w in words:
        conf, total = word_confidence(w)
        status_icon = "✅" if w["status"] == "verified" else "❓"
        conf_str = f"{conf:.0%}" if total > 0 else "—"

        with st.expander(
            f"{status_icon} **{w['canonical']}** — "
            f"hypothesis: *{w['meaning'] or '?'}* "
            f"({conf_str}, {total} tests)",
            expanded=False,
        ):
            if total < LOW_EVIDENCE_THRESHOLD:
                st.warning(f"⚠️ Low evidence — only {total} test(s) so far.")

            # Show variants
            variants = get_variants(conn, w["id"])
            if variants:
                st.write("**Variants:** " + ", ".join(variants))

            st.write(f"**Type:** {w['word_type'] or '—'}")
            st.write(f"**Person/Number:** {w['person_number'] or '—'}")
            st.write(f"**Status:** {w['status']}")
            st.write(f"**Confirmations:** {w.get('confirmations', 0)} "
                     f"| **Rejections:** {w.get('rejections', 0)}")

            # ── Edit meaning ──────────────────────────────────────
            new_meaning = st.text_input(
                "Update meaning hypothesis",
                value=w["meaning"] or "",
                key=f"wm_{w['id']}",
            ).strip()

            new_status = st.selectbox(
                "Status",
                ["unverified", "verified"],
                index=0 if w["status"] == "unverified" else 1,
                key=f"ws_{w['id']}",
            )

            new_variant = st.text_input(
                "Add a spelling variant",
                key=f"wv_{w['id']}",
                placeholder="e.g. eenu",
            ).strip().lower()

            if st.button("💾 Save changes", key=f"wsave_{w['id']}"):
                changes = {}
                if new_meaning != (w["meaning"] or ""):
                    changes["meaning"] = new_meaning or None
                if new_status != w["status"]:
                    changes["status"] = new_status
                if changes:
                    update_word(conn, w["id"], **changes)
                if new_variant and new_variant != w["canonical"]:
                    add_variant(conn, w["id"], new_variant)
                st.success("Saved.")
                st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: Sequences
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🔗 Sequences":
    st.title("🔗 Sequence Analysis")
    st.caption(
        "Upload or record a long clip. It will be split into word segments "
        "and each segment will be recognized. Meanings are *hypotheses*."
    )

    import matplotlib.pyplot as plt
    from sequences import (
        split_audio, extract_segment_features, recognize_sequence,
        count_ngrams, compare_phrase_vs_words,
    )
    from recognizer import Recognizer
    from rule_engine import apply_rules, suggest_for_gap, arrange_sentence

    # ── Build recognizer from known clips ──────────────────────────
    all_clips = get_clips(conn)
    clip_data = []
    for c in all_clips:
        feat = get_features(conn, c["id"])
        if feat:
            clip_data.append({
                "id": c["id"], "filename": c["filename"],
                "label": c["label"], "vector": feat["vector"],
            })

    recognizer = Recognizer()
    if len(clip_data) >= 2:
        recognizer.fit(
            [d["vector"] for d in clip_data],
            [d["label"] for d in clip_data],
        )

    # ── Upload audio ───────────────────────────────────────────────
    st.subheader("🎤 Input Audio")
    input_mode = st.radio("Source", ["Upload WAV file", "Record in browser"], horizontal=True)

    audio_bytes = None
    if input_mode == "Upload WAV file":
        uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
        if uploaded:
            audio_bytes = uploaded.getvalue()
    else:
        recorded = st.audio_input("🎤 Press to record a sequence")
        if recorded:
            audio_bytes = recorded.getvalue()

    top_db = st.slider(
        "Silence split threshold (top_db)",
        SPLIT_TOP_DB_MIN, SPLIT_TOP_DB_MAX, SPLIT_TOP_DB_DEFAULT,
        help="Lower = more aggressive splitting (detects quieter pauses as silence).",
    )

    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")

        if not recognizer.is_fitted:
            st.warning("Need at least 2 clips with features to recognize words. "
                       "Go to **Analysis** and analyze your clips first.")
            st.stop()

        # Save temp file for processing
        seq_dir = os.path.join(os.path.dirname(AUDIO_DIR), SEQUENCE_AUDIO_DIR_NAME)
        os.makedirs(seq_dir, exist_ok=True)
        import time
        tmp_path = os.path.join(seq_dir, f"seq_{int(time.time()*1000)}.wav")
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        # Split and recognize
        with st.spinner("Splitting audio into segments…"):
            segments = split_audio(tmp_path, top_db=top_db)

        if not segments:
            st.warning("No word segments found. Try lowering the split threshold.")
            st.stop()

        st.info(f"Found **{len(segments)}** segment(s).")

        # Get pending vectors for reuse
        pending_vecs = get_pending_sounds_with_vectors(conn)

        with st.spinner("Recognizing segments…"):
            results = recognize_sequence(segments, recognizer, pending_vecs)

        # ── Show results ───────────────────────────────────────────
        st.divider()
        st.subheader("📝 Recognized Sequence")

        labels = [r["label"] for r in results]
        recognized_text = " ".join(labels)
        st.markdown(f"### `{recognized_text}`")

        # ── Build word lookup ──────────────────────────────────────
        all_words = get_all_words(conn)
        word_lookup: dict[str, dict] = {}
        for w in all_words:
            word_lookup[w["canonical"]] = w
            # Also index variants
            for v in get_variants(conn, w["id"]):
                word_lookup[v] = w

        # ── Word-by-word meanings ──────────────────────────────────
        st.subheader("🔤 Word-by-Word Meanings")
        word_meanings = []
        rows = []
        for r in results:
            lbl = r["label"]
            info = word_lookup.get(lbl)
            meaning = info["meaning"] if info else None
            word_meanings.append(meaning)

            status = ""
            if r["is_unknown"]:
                status = "❓ Unknown"
            elif info:
                status = "✅" if info.get("status") == "verified" else "❓ Unverified"

            rows.append({
                "Word": lbl,
                "Meaning (hypothesis)": meaning or "?",
                "Status": status,
                "Distance": f"{r['distance']:.2f}",
                "Time": f"{r['start_sec']:.2f}–{r['end_sec']:.2f}s",
            })

        import pandas as pd_seq
        st.dataframe(pd_seq.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Phrase-level meaning ───────────────────────────────────
        st.subheader("📖 Phrase Meaning")
        phrase_text = " ".join(labels)
        saved_phrase = get_phrase_by_text(conn, phrase_text)

        if saved_phrase:
            st.success(
                f"**Known phrase:** *{saved_phrase['meaning'] or '?'}* "
                f"({saved_phrase['status']})"
            )
            comparison = compare_phrase_vs_words(
                saved_phrase["meaning"], word_meanings,
            )
            st.write(f"**Joined word meanings:** {comparison['joined_words']}")
            if comparison["match"]:
                st.info("✅ Phrase meaning matches joined word meanings.")
            else:
                st.warning("⚠️ Phrase meaning differs from joined word meanings.")
        else:
            st.warning("🆕 **New sentence found** — no saved phrase for this sequence.")

            # ── Sentence Arrangement & Rule-based suggestions ─────
            all_rules = get_all_rules(conn)
            arranged = arrange_sentence(labels, word_lookup, all_rules, saved_phrase=None)

            if arranged["ranked_suggestions"]:
                st.subheader("💡 Ranked Suggestions (Tentative)")
                for idx, s in enumerate(arranged["ranked_suggestions"]):
                    st.markdown(
                        f"**{idx + 1}.** `[tentative]` **{s['suggestion']}** "
                        f"— *{s['reason']}* (confidence: {s['confidence']:.0%})"
                    )

            if arranged["observations"]:
                st.subheader("📏 Rule Analysis")
                for obs in arranged["observations"]:
                    if obs.get("matched"):
                        st.success(f"✅ {obs['message']}")
                        if "english_reorder" in obs:
                            eng_words = obs["english_reorder"]
                            eng_meanings = [
                                word_lookup.get(w, {}).get("meaning", w)
                                for w in eng_words
                            ]
                            st.write(f"**English word order:** {' '.join(eng_meanings)}")
                    else:
                        st.info(f"ℹ️ {obs['message']}")

            # Suggestions for unknowns / gaps
            unknown_indices = [
                i for i, r in enumerate(results) if r["is_unknown"]
            ]
            if unknown_indices and all_rules:
                st.subheader("💡 Gap Analysis (Tentative Candidates)")
                for gi in unknown_indices:
                    suggs = suggest_for_gap(all_rules, labels, gi, word_lookup)
                    if suggs:
                        st.write(f"**Position {gi + 1}** (`{labels[gi]}`):")
                        for s in suggs[:5]:
                            info = word_lookup.get(s["candidate"], {})
                            meaning = info.get("meaning", "?")
                            pn = s.get("person_number_desc") or s.get("person_number", "")
                            pn_str = f" ({pn})" if pn else ""
                            st.write(
                                f"  → `[tentative]` **{s['candidate']}** = *{meaning}*{pn_str} "
                                f"— {s['reason']} (confidence: {s['confidence']:.0%})"
                            )
                    else:
                        st.write(f"**Position {gi + 1}** (`{labels[gi]}`): no rule suggestions.")

            # Initial default for phrase meaning input
            default_phrase_meaning = ""
            if arranged["ranked_suggestions"]:
                default_phrase_meaning = arranged["ranked_suggestions"][0]["suggestion"]
            else:
                default_phrase_meaning = arranged["wbw_joined"]

            # ── Right/Wrong on suggestion ──────────────────────────
            st.divider()
            st.subheader("✅ Confirm or Correct")
            st.caption("Right saves the phrase as a new verified entry. Wrong lets you type the correct meaning.")

            phrase_meaning_input = st.text_input(
                "Phrase meaning (confirm hypothesis or correct)",
                value=default_phrase_meaning,
                key="seq_phrase_meaning",
            )

            sc1, sc2 = st.columns(2)
            with sc1:
                if st.button("👍 Right — save phrase", key="seq_right", type="primary"):
                    pid = save_phrase(conn, phrase_text, phrase_meaning_input)
                    log_history(conn, "phrases", pid, None,
                                json.dumps({"action": "created_from_sequence",
                                            "text": phrase_text}))
                    # Save unknowns as pending
                    for r in results:
                        if r.get("is_unknown") and r.get("pending_name"):
                            save_pending_sound(
                                conn, r["pending_name"],
                                vector=r["vector"],
                            )
                    st.success(f"✅ Saved phrase: **{phrase_text}** = *{phrase_meaning_input}*")
                    st.rerun()
            with sc2:
                if st.button("👎 Wrong", key="seq_wrong"):
                    st.session_state["seq_show_correction"] = True

            if st.session_state.get("seq_show_correction"):
                correction = st.text_input(
                    "What is the correct meaning?",
                    key="seq_correction",
                )
                if st.button("Submit correction", key="seq_submit_corr"):
                    if correction:
                        pid = save_phrase(conn, phrase_text, correction)
                        log_history(conn, "phrases", pid, None,
                                    json.dumps({"action": "corrected_from_sequence",
                                                "text": phrase_text}))
                        for r in results:
                            if r.get("is_unknown") and r.get("pending_name"):
                                save_pending_sound(
                                    conn, r["pending_name"],
                                    vector=r["vector"],
                                )
                        st.session_state["seq_show_correction"] = False
                        st.success(f"✅ Saved corrected phrase: *{correction}*")
                        st.rerun()

        # ── N-gram bar charts ──────────────────────────────────────
        st.divider()
        st.subheader("📊 N-gram Frequencies")

        if len(labels) >= 2:
            bigrams = count_ngrams(labels, 2)
            if bigrams:
                fig_bi, ax_bi = plt.subplots(figsize=(8, 3))
                ax_bi.barh(list(bigrams.keys())[:15], list(bigrams.values())[:15])
                ax_bi.set_xlabel("Count")
                ax_bi.set_title("Bigrams")
                ax_bi.invert_yaxis()
                st.pyplot(fig_bi)
                plt.close(fig_bi)

        if len(labels) >= 3:
            trigrams = count_ngrams(labels, 3)
            if trigrams:
                fig_tri, ax_tri = plt.subplots(figsize=(8, 3))
                ax_tri.barh(list(trigrams.keys())[:15], list(trigrams.values())[:15])
                ax_tri.set_xlabel("Count")
                ax_tri.set_title("Trigrams")
                ax_tri.invert_yaxis()
                st.pyplot(fig_tri)
                plt.close(fig_tri)

        # ── Save sequence to DB ────────────────────────────────────
        seq_data = [
            {"label": r["label"], "distance": r["distance"],
             "is_unknown": r["is_unknown"],
             "start_sec": r["start_sec"], "end_sec": r["end_sec"]}
            for r in results
        ]
        save_sequence(conn, os.path.basename(tmp_path), seq_data)

    # ── Show past sequences ────────────────────────────────────────
    st.divider()
    st.subheader("📜 Past Sequences")
    past = get_all_sequences(conn)
    if past:
        for seq in past[:10]:
            labels_past = [s["label"] for s in seq["recognized"]]
            st.caption(
                f"`{seq['created_at']}` — {seq['filename']}: "
                f"**{' '.join(labels_past)}**"
            )
    else:
        st.info("No sequences analyzed yet.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: Rules
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📏 Rules":
    st.title("📏 Rule Management")
    st.caption(
        "Rules are data — word order patterns and pronoun-verb agreements. "
        "Each is a *hypothesis* until verified."
    )

    from rule_engine import all_plugins

    available_types = list(all_plugins().keys())

    # ── Add a new rule ─────────────────────────────────────────────
    st.subheader("➕ Add a Rule")

    if not available_types:
        st.warning("No rule plug-ins found. Add .py files to the rules/ directory.")
    else:
        rule_type = st.selectbox("Rule type", available_types, key="new_rule_type")

        if rule_type == "word_order":
            st.markdown("**Word Order Rule** — define the word-type pattern and English reorder.")
            wo_pattern = st.text_input(
                "Pattern (comma-separated word types)",
                placeholder="pronoun, question_word, verb",
                key="wo_pattern",
            )
            wo_english = st.text_input(
                "English order (comma-separated word types)",
                placeholder="question_word, verb, pronoun",
                key="wo_english",
            )
            wo_label = st.text_input("Label", placeholder="question sentence", key="wo_label")

            if st.button("💾 Save word order rule", key="save_wo"):
                pattern = [p.strip() for p in wo_pattern.split(",") if p.strip()]
                english = [p.strip() for p in wo_english.split(",") if p.strip()]
                if not pattern or not english:
                    st.warning("Fill in both pattern and English order.")
                else:
                    defn = {"pattern": pattern, "english_order": english, "label": wo_label}
                    rid = save_rule(conn, "word_order", defn)
                    log_history(conn, "rules", rid, None,
                                json.dumps({"action": "created", "type": "word_order"}))
                    st.success(f"✅ Saved word order rule #{rid}")
                    st.rerun()

        elif rule_type == "agreement":
            st.markdown("**Agreement Rule** — define pronoun ↔ verb form pairs.")
            n_pairs = st.number_input("Number of pairs", 1, 20, 3, key="ag_npairs")
            pairs = []
            for i in range(int(n_pairs)):
                c1, c2, c3 = st.columns(3)
                with c1:
                    pron = st.text_input(f"Pronoun #{i+1}", key=f"ag_pron_{i}")
                with c2:
                    verb = st.text_input(f"Verb form #{i+1}", key=f"ag_verb_{i}")
                with c3:
                    pn = st.text_input(f"Person/number #{i+1}", key=f"ag_pn_{i}",
                                       placeholder="e.g. 2sg")
                if pron.strip() and verb.strip():
                    pairs.append({
                        "pronoun": pron.strip().lower(),
                        "verb_form": verb.strip().lower(),
                        "person_number": pn.strip(),
                    })

            ag_label = st.text_input("Label", placeholder="pronoun-verb agreement",
                                      key="ag_label")

            if st.button("💾 Save agreement rule", key="save_ag"):
                if not pairs:
                    st.warning("Add at least one pronoun-verb pair.")
                else:
                    defn = {"pairs": pairs, "label": ag_label}
                    rid = save_rule(conn, "agreement", defn)
                    log_history(conn, "rules", rid, None,
                                json.dumps({"action": "created", "type": "agreement"}))
                    st.success(f"✅ Saved agreement rule #{rid}")
                    st.rerun()

        else:
            # Generic JSON editor for unknown rule types
            raw_json = st.text_area("Definition (JSON)", value="{}", key="gen_defn")
            if st.button("💾 Save rule", key="save_gen"):
                try:
                    defn = json.loads(raw_json)
                    rid = save_rule(conn, rule_type, defn)
                    log_history(conn, "rules", rid, None,
                                json.dumps({"action": "created", "type": rule_type}))
                    st.success(f"✅ Saved rule #{rid}")
                    st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")

    # ── List existing rules ────────────────────────────────────────
    st.divider()
    st.subheader("📋 Existing Rules")
    rules = get_all_rules(conn)
    if not rules:
        st.info("No rules yet.")
    else:
        for rule in rules:
            plugin_cls = all_plugins().get(rule["rule_type"])
            desc = plugin_cls.describe(rule["definition"]) if plugin_cls else str(rule["definition"])
            status_icon = "✅" if rule["status"] == "verified" else "❓"

            with st.expander(
                f"{status_icon} Rule #{rule['id']}: {rule['rule_type']} — {desc}",
                expanded=False,
            ):
                st.write(f"**Status:** {rule['status']}")
                st.write(f"**Created:** {rule['created_at']}")

                edit_def_str = st.text_area(
                    "Definition (JSON data)",
                    value=json.dumps(rule["definition"], indent=2),
                    key=f"rdef_{rule['id']}",
                    height=120,
                )

                new_status = st.selectbox(
                    "Status", ["unverified", "verified"],
                    index=0 if rule["status"] == "unverified" else 1,
                    key=f"rstatus_{rule['id']}",
                )

                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("💾 Save changes", key=f"rupdate_{rule['id']}"):
                        try:
                            parsed_def = json.loads(edit_def_str)
                            update_rule(conn, rule["id"], definition=parsed_def, status=new_status)
                            st.success("Rule updated.")
                            st.rerun()
                        except json.JSONDecodeError as err:
                            st.error(f"Invalid JSON: {err}")
                with rc2:
                    if st.button("🗑️ Delete", key=f"rdelete_{rule['id']}"):
                        log_history(conn, "rules", rule["id"],
                                    json.dumps(rule["definition"]),
                                    json.dumps({"action": "deleted"}))
                        delete_rule(conn, rule["id"])
                        st.success("Deleted.")
                        st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: Test  (Right / Wrong confirmation flow)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "✅ Test":
    st.title("✅ Test & Confirm Hypotheses")
    st.caption(
        "Listen to a clip and confirm or reject the recognizer's hypothesis. "
        "This builds confidence scores."
    )

    words = get_all_words(conn)
    if not words:
        st.info("No words in the dictionary yet. Create some from the Pending page.")
        st.stop()

    # Build recognizer from all clips with features
    all_clips = get_clips(conn)
    clip_data = []
    for c in all_clips:
        feat = get_features(conn, c["id"])
        if feat:
            clip_data.append({
                "id": c["id"], "filename": c["filename"],
                "label": c["label"], "vector": feat["vector"],
            })

    if len(clip_data) < 2:
        st.warning("Need at least 2 clips with features to test recognition.")
        st.stop()

    from recognizer import Recognizer
    recognizer = Recognizer()
    recognizer.fit([d["vector"] for d in clip_data],
                   [d["label"] for d in clip_data])

    if not recognizer.is_fitted:
        st.warning("Recognizer could not be fitted.")
        st.stop()

    # Pick a clip to test
    filenames = [d["filename"] for d in clip_data]
    sel = st.selectbox("Pick a clip to test", filenames, key="test_clip")

    if sel:
        idx = filenames.index(sel)
        clip = clip_data[idx]
        wav = load_wav(clip["filename"])
        if wav:
            st.audio(wav, format="audio/wav")

        result = recognizer.recognize(clip["vector"])

        st.subheader("Recognition Result")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.metric("Predicted label", result["label"])
        with rc2:
            st.metric("Distance", f"{result['distance']:.2f}")

        # Look up the word
        word = lookup_by_variant(conn, result["label"])
        if word:
            conf, total = word_confidence(word)
            is_unverified = word.get("status") == "unverified"
            status_badge = "⚠️ Unverified (Low Confidence Suggestion)" if is_unverified else "✅ Verified"

            st.write(
                f"**Meaning hypothesis:** *{word['meaning'] or '?'}* "
                f"[{status_badge}]"
            )
            if is_unverified:
                st.info("ℹ️ This word hypothesis is **unverified** — treat as tentative, low-confidence suggestion.")
            elif total < LOW_EVIDENCE_THRESHOLD:
                st.warning(f"⚠️ Low evidence — only {total} test(s).")

            # Right / Wrong buttons
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                if st.button("👍 Right", key="right_btn", type="primary"):
                    confirm_word(conn, word["id"])
                    st.success("✅ Confirmed! Confidence updated.")
                    st.rerun()
            with bcol2:
                if st.button("👎 Wrong", key="wrong_btn"):
                    st.session_state["show_correction"] = True

            if st.session_state.get("show_correction"):
                correction = st.text_input(
                    "What is the correct text or meaning?",
                    key="correction_input",
                    placeholder="e.g. 'cat' not 'dog'",
                )
                if st.button("Submit correction", key="submit_corr"):
                    reject_word(conn, word["id"], correction=correction)
                    st.session_state["show_correction"] = False
                    st.warning("❌ Rejected. Correction recorded.")
                    st.rerun()
        elif result["is_unknown"]:
            st.info("This clip was not recognized (unknown).")
        else:
            st.info(
                f"Label **{result['label']}** is not yet in the word dictionary. "
                f"Create it from the Pending page."
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📋 Summary":
    st.title("📋 Hypothesis Summary")
    st.caption("All words with their confidence and status.")

    words = get_all_words(conn)
    if not words:
        st.info("No words yet.")
        st.stop()

    rows = []
    for w in words:
        conf, total = word_confidence(w)
        variants = get_variants(conn, w["id"])
        rows.append({
            "Word": w["canonical"],
            "Variants": ", ".join(variants) if variants else "—",
            "Meaning (hypothesis)": w["meaning"] or "?",
            "Type": w["word_type"] or "—",
            "Status": w["status"],
            "Tests": total,
            "Confidence": f"{conf:.0%}" if total > 0 else "—",
            "⚠️": "low evidence" if total < LOW_EVIDENCE_THRESHOLD else "",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Recent history ─────────────────────────────────────────────
    st.divider()
    st.subheader("📜 Recent Changes")
    history = get_recent_history(conn, limit=15)
    if history:
        for h in history:
            st.caption(
                f"`{h['ts']}` — **{h['table_name']}** #{h['row_id']}: "
                f"{h.get('new_value', '')}"
            )
    else:
        st.info("No history yet.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📈 Evaluation":
    import matplotlib.pyplot as plt

    st.title("📈 Model Evaluation & Performance")
    st.caption(
        "Speaker-wise holdout evaluation, classifier comparison, noise robustness, "
        "and honest reporting on model limitations."
    )

    col_btn, col_exp = st.columns([1, 1])
    with col_btn:
        run_eval = st.button("🚀 Run Evaluation", type="primary")

    if run_eval or "eval_result" in st.session_state:
        if run_eval or "eval_result" not in st.session_state:
            with st.spinner("Running evaluation across classifiers and noise levels..."):
                st.session_state.eval_result = run_evaluation(conn)

        res = st.session_state.eval_result

        # ── Warnings ───────────────────────────────────────────────
        if res.get("warnings"):
            for w in res["warnings"]:
                st.warning(f"⚠️ {w}")

        # ── Dataset Overview ───────────────────────────────────────
        st.subheader("📊 Dataset Overview")
        ds = res.get("dataset", {})
        vocab = res.get("vocabulary", {})
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Clips", ds.get("n_clips", 0))
        c2.metric("Labels", ds.get("n_labels", 0))
        c3.metric("Speakers", ds.get("n_speakers", 0))
        c4.metric("Words (Vocab)", vocab.get("n_words", 0))
        c5.metric("Phrases", vocab.get("n_phrases", 0))
        c6.metric("Pending Sounds", vocab.get("n_pending", 0))

        if "split_method" in res:
            sizes = res.get("split_sizes", {})
            st.info(
                f"**Split Strategy:** {res['split_method']} "
                f"(Train: {sizes.get('train', '?')} clips, Test: {sizes.get('test', '?')} clips)"
            )

        # ── Classifier Comparison ──────────────────────────────────
        st.divider()
        st.subheader("🤖 Classifier Comparison")
        st.caption("Comparing pure-numpy classifiers on identical extracted features.")

        classifiers = res.get("classifiers", {})
        if classifiers:
            clf_rows = []
            best_clf_name = None
            best_acc = -1.0
            for name, metrics in classifiers.items():
                if "error" in metrics:
                    clf_rows.append({"Classifier": name, "Accuracy": f"Error: {metrics['error']}", "Status": "Failed"})
                else:
                    acc = metrics.get("accuracy", 0.0)
                    if acc > best_acc:
                        best_acc = acc
                        best_clf_name = name
                    clf_rows.append({
                        "Classifier": name,
                        "Accuracy": f"{acc:.1%}",
                        "Test Samples": metrics.get("n_total", 0),
                        "Classes": len(metrics.get("labels", [])),
                    })
            st.dataframe(pd.DataFrame(clf_rows), use_container_width=True, hide_index=True)

            # Per-class metrics & Confusion Matrix for best / primary classifier
            target_clf = best_clf_name or (list(classifiers.keys())[0] if classifiers else None)
            if target_clf and "error" not in classifiers[target_clf]:
                t_metrics = classifiers[target_clf]
                st.markdown(f"#### Detailed Performance: **{target_clf}**")

                col_pc, col_cm = st.columns([1, 1])
                with col_pc:
                    st.markdown("**Per-Class Precision & Recall**")
                    pc_dict = t_metrics.get("per_class", {})
                    if pc_dict:
                        pc_rows = []
                        for lbl, stats in pc_dict.items():
                            pc_rows.append({
                                "Label": lbl,
                                "Precision": f"{stats['precision']:.1%}",
                                "Recall": f"{stats['recall']:.1%}",
                                "Support": stats["support"],
                            })
                        st.dataframe(pd.DataFrame(pc_rows), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No per-class breakdown available.")

                with col_cm:
                    st.markdown("**Confusion Matrix Heatmap**")
                    cm = np.array(t_metrics.get("confusion_matrix", []))
                    labels = t_metrics.get("labels", [])
                    if cm.size > 0 and len(labels) > 0:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
                        ax.set_xticks(range(len(labels)))
                        ax.set_yticks(range(len(labels)))
                        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
                        ax.set_yticklabels(labels, fontsize=9)
                        ax.set_xlabel("Predicted", fontsize=10)
                        ax.set_ylabel("True", fontsize=10)
                        max_val = cm.max() if cm.max() > 0 else 1
                        for i in range(len(labels)):
                            for j in range(len(labels)):
                                ax.text(
                                    j, i, str(cm[i, j]),
                                    ha="center", va="center",
                                    color="white" if cm[i, j] > max_val / 2 else "black",
                                    fontsize=9,
                                )
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.caption("No confusion matrix data.")

        # ── Cross-Speaker Accuracy ─────────────────────────────────
        if res.get("cross_speaker"):
            st.divider()
            st.subheader("👥 Cross-Speaker Generalisation")
            st.caption("Training on all other speakers and testing exclusively on the held-out speaker.")
            cs_rows = []
            for spk, info in res["cross_speaker"].items():
                acc = info.get("accuracy")
                cs_rows.append({
                    "Held-Out Speaker": spk,
                    "Accuracy": f"{acc:.1%}" if acc is not None else "N/A",
                    "Test Samples": info.get("n_test", "—"),
                    "Note": info.get("note", "OK"),
                })
            st.dataframe(pd.DataFrame(cs_rows), use_container_width=True, hide_index=True)

        # ── Noise Robustness ───────────────────────────────────────
        if res.get("noise_robustness"):
            st.divider()
            st.subheader("🔊 Noise Robustness (Synthetic SNR Degradation)")
            st.caption("Model trained on clean signals, tested against feature vectors with synthetic noise.")
            nr_data = res["noise_robustness"]
            nr_df = pd.DataFrame([
                {"SNR Level": k, "Accuracy": v} for k, v in nr_data.items()
            ])
            st.bar_chart(nr_df.set_index("SNR Level"))

        # ── User Confirmation Stats ────────────────────────────────
        if res.get("confirmation"):
            st.divider()
            st.subheader("👍 User Feedback & Confirmations")
            conf = res["confirmation"]
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Total Tests", conf.get("total_tests", 0))
            f2.metric("Confirmed (Right)", conf.get("total_right", 0))
            f3.metric("Rejected (Wrong)", conf.get("total_wrong", 0))
            f4.metric("Agreement Share", f"{conf.get('right_share', 0.0):.1%}")

        # ── Export Report & Limitations ────────────────────────────
        st.divider()
        st.subheader("📄 Evaluation Report & Limitations")

        report_md = generate_report(res)

        col_exp_btn, col_dl_btn = st.columns([1, 1])
        with col_exp_btn:
            if st.button("📥 Save Report to Disk"):
                saved_path = export_report(conn)
                st.success(f"Report saved to `{saved_path}`")

        with col_dl_btn:
            st.download_button(
                label="⬇️ Download Report (.md)",
                data=report_md,
                file_name=f"ucs_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )

        with st.expander("📖 View Limitations Section (auto-included in all reports)", expanded=True):
            st.markdown(LIMITATIONS_TEXT)
    else:
        st.info("Click **Run Evaluation** above to compute metrics, compare classifiers, and inspect noise robustness.")

