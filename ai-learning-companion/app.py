import os
import json
import shutil
from datetime import datetime
import streamlit as st

# Whisper STT
try:
    import whisper
except Exception:
    whisper = None

from utils.text_cleaner import clean_transcript
from llm.summarizer import generate_notes
from llm.flashcards import generate_flashcards
from llm.quiz_generator import generate_quiz
from llm.concept_identifier import identify_concepts, filter_concepts
from rag.embeddings import EmbeddingModel
from rag.vector_store import LectureVectorStore
from rag.chat import answer_question


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIRS = {
    "audio": os.path.join(PROJECT_ROOT, "audio"),
    "raw": os.path.join(PROJECT_ROOT, "transcripts", "raw"),
    "cleaned": os.path.join(PROJECT_ROOT, "transcripts", "cleaned"),
    "rag_index": os.path.join(PROJECT_ROOT, "rag", "faiss_index"),
}


def ensure_dirs():
    for d in DATA_DIRS.values():
        os.makedirs(d, exist_ok=True)


def transcribe_audio(audio_path: str, language: str = "auto") -> str:
    """Transcribe audio to text using Whisper.
    - For .wav files: decode in Python and pass waveform to Whisper (no ffmpeg).
    - For other formats: require ffmpeg (or convert to .wav).
    - Optional `language` code to override auto-detection (e.g., 'en').
    """
    if whisper is None:
        raise RuntimeError(
            "Whisper is not installed. Please install via requirements.txt and try again."
        )

    ext = os.path.splitext(audio_path)[1].lower()
    model = whisper.load_model("base")

    if ext == ".wav":
        try:
            import numpy as np
            from scipy.io import wavfile
            from scipy import signal

            sr, data = wavfile.read(audio_path)
            # Normalize amplitude to [-1, 1] based on original dtype
            import numpy as np
            if np.issubdtype(data.dtype, np.integer):
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.uint8:
                    data = (data.astype(np.float32) - 128.0) / 128.0
                else:
                    max_val = float(np.max(np.abs(data))) or 1.0
                    data = data.astype(np.float32) / max_val
            else:
                data = data.astype(np.float32)
                max_abs = float(np.max(np.abs(data)))
                if max_abs > 1.0:
                    data = data / max_abs
            # Stereo to mono
            if len(data.shape) == 2 and data.shape[1] > 1:
                data = data.mean(axis=1)

            target_sr = 16000
            if sr != target_sr:
                # Resample to 16k using polyphase filtering
                data = signal.resample_poly(data, target_sr, sr).astype(np.float32)

            # Transcribe directly from ndarray; optionally fix language
            if language and language.lower() != "auto":
                result = whisper.transcribe(model, data, language=language.lower(), fp16=False)
            else:
                result = whisper.transcribe(model, data, fp16=False)
            return result.get("text", "").strip()
        except Exception as e:
            # Fallback to standard path (likely requires ffmpeg)
            try:
                result = model.transcribe(audio_path)
                return result.get("text", "").strip()
            except Exception as inner:
                raise RuntimeError(
                    f"WAV decoding failed: {e}. Fallback also failed: {inner}."
                )
    else:
        # For MP3 and other formats, Whisper needs ffmpeg. Provide friendly guidance if missing.
        import shutil
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "FFmpeg is required for non-WAV formats. Install FFmpeg and ensure it's in PATH, "
                "or convert your audio to .wav and re-upload."
            )
        if language and language.lower() != "auto":
            result = model.transcribe(audio_path, language=language.lower(), fp16=False)
        else:
            result = model.transcribe(audio_path, fp16=False)
        return result.get("text", "").strip()


def save_text(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def load_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_index_for_lecture(lecture_id: str, cleaned_text: str, notes_md: str):
    """Chunk cleaned transcript and notes, embed, and store in FAISS."""
    embedder = EmbeddingModel()
    store = LectureVectorStore(index_dir=os.path.join(DATA_DIRS["rag_index"], lecture_id))
    texts = []
    # Prefer LangChain splitter; fallback to simple chunking
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
        )
        if cleaned_text:
            texts.extend(splitter.split_text(cleaned_text))
        if notes_md:
            texts.extend(splitter.split_text(notes_md))
    except Exception:
        def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            chunks = []
            for p in paragraphs:
                start = 0
                while start < len(p):
                    end = min(start + chunk_size, len(p))
                    chunks.append(p[start:end])
                    start = max(end - overlap, end)
            return chunks
        texts.extend(chunk_text(cleaned_text))
        texts.extend(chunk_text(notes_md))

    if not texts:
        raise ValueError("No text available to index.")

    embeddings = embedder.embed_texts(texts)
    store.build(texts=texts, embeddings=embeddings)
    store.save()
    return store


def init_session_state():
    if "lecture_id" not in st.session_state:
        st.session_state.lecture_id = None
    if "raw_transcript" not in st.session_state:
        st.session_state.raw_transcript = ""
    if "cleaned_transcript" not in st.session_state:
        st.session_state.cleaned_transcript = ""
    if "notes_md" not in st.session_state:
        st.session_state.notes_md = ""
    if "flashcards" not in st.session_state:
        st.session_state.flashcards = []
    if "quiz" not in st.session_state:
        st.session_state.quiz = {"mcq": [], "short": []}
    if "concepts" not in st.session_state:
        st.session_state.concepts = []


def main():
    st.set_page_config(page_title="LectureMind AI", layout="wide")
    ensure_dirs()
    init_session_state()
    ffmpeg_available = shutil.which("ffmpeg") is not None

    st.title("LectureMind AI — AI-powered Learning Companion")
    st.caption("Upload lectures, auto-generate study materials, and chat with your content.")

    # Sidebar: Lecture selection and state
    with st.sidebar:
        st.header("Lecture Session")
        current_lecture = st.text_input("Lecture ID (auto if empty)", value=st.session_state.lecture_id or "")
        if st.button("Set Lecture ID"):
            st.session_state.lecture_id = current_lecture.strip() or datetime.now().strftime("lecture_%Y%m%d_%H%M%S")
            st.success(f"Current lecture ID: {st.session_state.lecture_id}")

        st.divider()
        st.write("Storage Paths:")
        if st.session_state.lecture_id:
            st.code(os.path.join(DATA_DIRS["audio"], f"{st.session_state.lecture_id}.wav"))
            st.code(os.path.join(DATA_DIRS["raw"], f"{st.session_state.lecture_id}.txt"))
            st.code(os.path.join(DATA_DIRS["cleaned"], f"{st.session_state.lecture_id}.txt"))

        st.divider()
        # --- Sidebar AI Settings ---
        st.header("🤖 AI Settings")
        ai_mode = st.radio("AI Source", ["Local AI (Gemma 3)", "Cloud AI (OpenAI)"], index=0)
        
        if ai_mode == "Cloud AI (OpenAI)":
            api_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
            api_base = st.text_input("API Base URL", value="https://api.openai.com/v1")
            model_name = st.text_input("Model Name", value="gpt-3.5-turbo")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                os.environ["OPENAI_API_BASE"] = api_base
                os.environ["OPENAI_MODEL"] = model_name
        else:
            # Local AI Settings (Gemma via Ollama)
            local_base = st.sidebar.text_input("Local API URL", value="http://localhost:11434/v1")
            local_model = st.sidebar.text_input("Local Model", value="gemma3:1b")
            os.environ["OPENAI_API_BASE"] = local_base
            os.environ["OPENAI_MODEL"] = local_model
            
            # Test Connection Button
            if st.button("🔌 Test Local AI Connection"):
                try:
                    import requests
                    # Try to list models from Ollama
                    resp = requests.get(local_base.replace("/v1", "/api/tags"), timeout=5)
                    if resp.status_code == 200:
                        models = [m['name'] for m in resp.json().get('models', [])]
                        if local_model in models or any(local_model in m for m in models):
                            st.success(f"✅ Connected! Found: {local_model}")
                        else:
                            st.warning(f"⚠️ Connected, but model '{local_model}' not found in: {models}")
                    else:
                        st.error(f"❌ Connection failed (Status: {resp.status_code})")
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")

            # Clear API key for local mode to avoid confusion
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

        st.divider()
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()

    # 1) Audio Upload
    st.header("1️⃣ Audio Upload")
    upload_types = ["wav", "mp3"] if ffmpeg_available else ["wav"]
    uploaded = st.file_uploader(
        "Upload lecture audio (.mp3/.wav)" if ffmpeg_available else "Upload lecture audio (.wav only)",
        type=upload_types,
    )
    if not ffmpeg_available:
        st.info("MP3 requires FFmpeg. Install FFmpeg (add to PATH) or upload a .wav file.")
    if uploaded:
        if not st.session_state.lecture_id:
            # Auto-assign Lecture ID so users can upload immediately
            st.session_state.lecture_id = datetime.now().strftime("lecture_%Y%m%d_%H%M%S")
            try:
                st.toast(f"Lecture ID auto-set: {st.session_state.lecture_id}")
            except Exception:
                st.info(f"Lecture ID auto-set: {st.session_state.lecture_id}")
        audio_ext = os.path.splitext(uploaded.name)[1].lower()
        audio_path = os.path.join(DATA_DIRS["audio"], f"{st.session_state.lecture_id}{audio_ext}")
        with open(audio_path, "wb") as f:
            f.write(uploaded.read())
        st.success(f"Saved: {audio_path}")

    # 2) Speech-to-Text
    st.header("2️⃣ Speech-to-Text (Whisper)")
    # Language selector to avoid wrong auto-detection
    lang_options = {
        "Auto": "auto",
        "English (en)": "en",
        "Chinese (zh)": "zh",
        "Spanish (es)": "es",
        "French (fr)": "fr",
        "German (de)": "de",
        "Japanese (ja)": "ja",
        "Korean (ko)": "ko",
        "Hindi (hi)": "hi",
        "Portuguese (pt)": "pt",
    }
    stt_lang_label = st.selectbox("Transcription language", list(lang_options.keys()), index=0)
    stt_language = lang_options[stt_lang_label]
    if st.session_state.lecture_id:
        audio_candidates = [
            os.path.join(DATA_DIRS["audio"], f"{st.session_state.lecture_id}.wav"),
            os.path.join(DATA_DIRS["audio"], f"{st.session_state.lecture_id}.mp3"),
        ]
        audio_path = next((p for p in audio_candidates if os.path.exists(p)), None)
        if audio_path:
            needs_ffmpeg = audio_path.lower().endswith(".mp3") and not ffmpeg_available
            if needs_ffmpeg:
                st.error("This lecture is saved as .mp3, but FFmpeg is not available. Install FFmpeg or upload .wav.")
            if st.button("Transcribe Audio", disabled=needs_ffmpeg):
                try:
                    raw_text = transcribe_audio(audio_path, language=stt_language)
                    st.session_state.raw_transcript = raw_text
                    raw_path = os.path.join(DATA_DIRS["raw"], f"{st.session_state.lecture_id}.txt")
                    save_text(raw_path, raw_text)
                    st.success(f"Transcript saved to: {raw_path}")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
        else:
            st.info("Upload audio first. Supported: .wav, .mp3")

    # 3) Transcript Cleaning
    st.header("3️⃣ Transcript Cleaning")
    raw_display = st.text_area("Raw Transcript", value=st.session_state.raw_transcript, height=200)
    if st.button("Clean Transcript"):
        try:
            cleaned = clean_transcript(raw_display)
            st.session_state.cleaned_transcript = cleaned
            cleaned_path = os.path.join(DATA_DIRS["cleaned"], f"{st.session_state.lecture_id}.txt")
            save_text(cleaned_path, cleaned)
            st.success(f"Cleaned transcript saved to: {cleaned_path}")
        except Exception as e:
            st.error(f"Cleaning failed: {e}")

    # 4) Concept Identification
    st.header("4️⃣ Concept Identification & Filtering")
    if st.button("Identify Key Concepts"):
        try:
            raw_concepts = identify_concepts(st.session_state.cleaned_transcript or st.session_state.raw_transcript)
            filtered = filter_concepts(raw_concepts)
            st.session_state.concepts = filtered
            st.success(f"Identified concepts: {', '.join(filtered)}")
        except Exception as e:
            st.error(f"Concept identification failed: {e}")
    
    st.session_state.concepts = st.multiselect("Active Concepts (Editable)", options=st.session_state.concepts, default=st.session_state.concepts)

    # 5) AI Content Generation
    st.header("5️⃣ AI Content Generation")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Generate Notes"):
            try:
                notes_md = generate_notes(st.session_state.cleaned_transcript or st.session_state.raw_transcript, concepts=st.session_state.concepts)
                st.session_state.notes_md = notes_md
                notes_path = os.path.join(PROJECT_ROOT, "llm", "outputs", f"{st.session_state.lecture_id}_notes.md")
                save_text(notes_path, notes_md)
                st.success(f"Notes saved to: {notes_path}")
            except Exception as e:
                st.error(f"Notes generation failed: {e}")
    with col2:
        if st.button("Generate Flashcards"):
            try:
                cards = generate_flashcards(st.session_state.cleaned_transcript or st.session_state.raw_transcript, concepts=st.session_state.concepts)
                st.session_state.flashcards = cards
                fc_path = os.path.join(PROJECT_ROOT, "llm", "outputs", f"{st.session_state.lecture_id}_flashcards.json")
                os.makedirs(os.path.dirname(fc_path), exist_ok=True)
                with open(fc_path, "w", encoding="utf-8") as f:
                    json.dump(cards, f, ensure_ascii=False, indent=2)
                st.success(f"Flashcards saved to: {fc_path}")
            except Exception as e:
                st.error(f"Flashcards generation failed: {e}")
    with col3:
        if st.button("Generate Quiz"):
            try:
                quiz = generate_quiz(st.session_state.cleaned_transcript or st.session_state.raw_transcript, concepts=st.session_state.concepts)
                st.session_state.quiz = quiz
                quiz_path = os.path.join(PROJECT_ROOT, "llm", "outputs", f"{st.session_state.lecture_id}_quiz.json")
                os.makedirs(os.path.dirname(quiz_path), exist_ok=True)
                with open(quiz_path, "w", encoding="utf-8") as f:
                    json.dump(quiz, f, ensure_ascii=False, indent=2)
                st.success(f"Quiz saved to: {quiz_path}")
            except Exception as e:
                st.error(f"Quiz generation failed: {e}")

    with st.expander("Notes (Markdown)"):
        st.markdown(st.session_state.notes_md or "No notes yet.")
    with st.expander("Flashcards"):
        if st.session_state.flashcards:
            for i, card in enumerate(st.session_state.flashcards, 1):
                st.markdown(f"**Q{i}:** {card['question']}")
                st.write(f"A{i}: {card['answer']}")
                st.divider()
        else:
            st.write("No flashcards yet.")
    with st.expander("Quiz"):
        if st.session_state.quiz["mcq"] or st.session_state.quiz["short"]:
            st.subheader("MCQs")
            for q in st.session_state.quiz["mcq"]:
                st.write(f"- {q['question']}")
                for opt in q["options"]:
                    st.write(f"  • {opt}")
                st.write(f"Answer: {q['answer']}")
            st.subheader("Short Answer")
            for q in st.session_state.quiz["short"]:
                st.write(f"- {q['question']}")
                st.write(f"Expected: {q['answer']}")
        else:
            st.write("No quiz yet.")

    # 6) RAG Pipeline - Build Index
    st.header("6️⃣ RAG Pipeline — Build FAISS Index")
    if st.button("Build/Search Index"):
        try:
            store = build_index_for_lecture(
                st.session_state.lecture_id,
                st.session_state.cleaned_transcript or st.session_state.raw_transcript,
                st.session_state.notes_md or "",
            )
            st.success("FAISS index built and saved.")
        except Exception as e:
            st.error(f"Index build failed: {e}")

    # 7) Chat with Lecture
    st.header("7️⃣ Chat with Lecture")
    question = st.text_input("Ask a question about this lecture:")
    if st.button("Answer Question") and question.strip():
        try:
            store = LectureVectorStore(index_dir=os.path.join(DATA_DIRS["rag_index"], st.session_state.lecture_id))
            if not store.exists():
                st.error("No index found. Please build the index first.")
            else:
                store.load()
                answer = answer_question(question, store)
                st.markdown(f"**Answer:**\n\n{answer}")
        except Exception as e:
            st.error(f"Chat failed: {e}")

    st.header("8️⃣ Transcript Viewer")
    st.text_area("Cleaned Transcript", value=st.session_state.cleaned_transcript, height=300)


if __name__ == "__main__":
    main()
