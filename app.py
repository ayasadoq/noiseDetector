import io
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import streamlit as st
from resemblyzer import VoiceEncoder, preprocess_wav

SR = 16000
VOICES_FILE = "voices.npz"

st.set_page_config(page_title="Sound detector", layout="wide")


@st.cache_resource
def encoder():
    return VoiceEncoder()


def load_audio(uploaded):
    data, sr = sf.read(io.BytesIO(uploaded.getvalue()))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return data.astype(np.float32)


def dbfs(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def embed(wav):
    return encoder().embed_utterance(preprocess_wav(wav, source_sr=SR))


def save_voices():
    np.savez(VOICES_FILE, **{k: np.array(v) for k, v in st.session_state.voices.items()})


if "voices" not in st.session_state:
    try:
        d = np.load(VOICES_FILE)
        st.session_state.voices = {k: list(d[k]) for k in d.files}
    except Exception:
        st.session_state.voices = {}


def classify(emb, threshold, margin):
    sims = {n: float(np.dot(emb, np.mean(e, axis=0) / np.linalg.norm(np.mean(e, axis=0))))
            for n, e in st.session_state.voices.items()}
    ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
    best, score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else -1
    if score < threshold or (score - second) < margin:
        return "Inconclusive", score, sims
    return best, score, sims


# ---------------- Sidebar: enrollment ----------------
with st.sidebar:
    st.header("1. Register a person")
    name = st.text_input("ID (e.g. H1, H2, H3)")
    rec = st.audio_input("Record a reference voice (10-15 s)", key="enroll_mic")
    up = st.file_uploader("Or upload a WAV", type=["wav"], key="enroll_up")
    if st.button("Add this voice"):
        src = rec or up
        if not name or src is None:
            st.warning("Enter an ID and provide audio.")
        else:
            try:
                st.session_state.voices.setdefault(name, []).append(embed(load_audio(src)))
                save_voices()
                st.success(f"Voice added for {name}")
            except Exception as e:
                st.error(f"Audio too short or unusable: {e}")
    st.write("Registered:", ", ".join(st.session_state.voices) or "none")
    if st.button("Reset all voices"):
        st.session_state.voices = {}
        save_voices()
        st.rerun()

    st.header("Settings")
    threshold = st.slider("Similarity threshold", 0.50, 0.95, 0.75, 0.01)
    margin = st.slider("Min gap vs 2nd best", 0.00, 0.20, 0.05, 0.01)
    silence_db = st.slider("Silence level (dBFS)", -80, -20, -50)
    win = st.slider("Window (s)", 1.0, 5.0, 3.0, 0.5)

# ---------------- Main: analysis ----------------
st.title("Classroom sound detector and analysis")
st.caption("PC prototype · relative dBFS level · experimental speaker recognition")
st.header("2. Analyse a recording")

mic = st.audio_input("Speak into the mic")
wav_up = st.file_uploader("Or provide a WAV file", type=["wav"], key="an_up")
src = mic or wav_up

if src is not None:
    if not st.session_state.voices:
        st.warning("Register at least one person first.")
        st.stop()

    audio = load_audio(src)
    step = int(win * SR)
    rows = []
    for i in range(0, max(len(audio) - step // 2, 1), step):
        seg = audio[i:i + step]
        level = dbfs(seg)
        t = f"{i / SR:.1f}s"
        if level < silence_db:
            rows.append({"time": t, "dBFS": level, "speaker": "Silence", "score": None})
            continue
        try:
            who, score, _ = classify(embed(seg), threshold, margin)
        except Exception:
            who, score = "Inconclusive", None
        rows.append({"time": t, "dBFS": level, "speaker": who, "score": score})

    df = pd.DataFrame(rows)
    speech = df[df.speaker != "Silence"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Duration", f"{len(audio) / SR:.1f} s")
    c2.metric("Average level", f"{dbfs(audio):.1f} dBFS")
    if len(speech):
        top = speech.speaker.value_counts().idxmax()
        c3.metric("Main result", top)
    else:
        c3.metric("Main result", "Silence")

    st.subheader("Timeline")
    st.dataframe(df, use_container_width=True)
    st.subheader("Level (dBFS)")
    st.line_chart(df.set_index("time")["dBFS"])
    if len(speech):
        st.subheader("Speaking share")
        st.bar_chart(speech.speaker.value_counts())
