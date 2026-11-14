"""
UCS — Universal Communication System
Streamlit app (UI only). Logic lives in db.py, audio_io.py, features.py,
clustering.py, recognizer.py.
"""
import os
import json

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
)
from audio_io import (
    build_filename,
    save_wav_from_file,
    load_wav,
    parse_filename,
    list_recordings,
)
from config import (
    AUDIO_DIR, FEATURE_SR, HOP_LENGTH, N_FFT,
    MIN_CLIPS_FOR_CLUSTERING, DBSCAN_EPS_DEFAULT, DBSCAN_MIN_SAMPLES_DEFAULT,
    RECOGNIZE_THRESHOLD, LOW_EVIDENCE_THRESHOLD, WORD_TYPES,
    SPLIT_TOP_DB_DEFAULT, SPLIT_TOP_DB_MIN, SPLIT_TOP_DB_MAX,
    MIN_SEGMENT_DURATION, SEQUENCE_AUDIO_DIR_NAME,
)

# ── Initialise DB once per session ─────────────────────────────────
if "conn" not in st.session_state:
    st.session_state.conn = init_db()

conn = st.session_state.conn

# ── Sidebar navigation ────────────────────────────────────────────
st.set_page_config(page_title="UCS — Universal Communication System", layout="wide")

PAGES = [
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
#  PAGE: Record
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == "🎙️ Record":
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
