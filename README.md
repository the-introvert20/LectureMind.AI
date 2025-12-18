# LectureMind AI

LectureMind AI is an end-to-end, local-first web app that turns lecture audio into structured study materials and enables Retrieval-Augmented Generation (RAG) chat with your lectures.

This repository contains the main project under `ai-learning-companion/`. A copy of the detailed README also exists inside that folder. This top-level README ensures GitHub displays project info on the front page.

## Quick Start
- Create/activate your Python venv.
- Install dependencies:
  - `pip install -r ai-learning-companion/requirements.txt`
- Run the app:
  - `streamlit run ai-learning-companion/app.py`

## Features
- Upload `.mp3`/`.wav` lecture audio
- Transcribe using Whisper (open-source)
- Clean transcript (remove fillers, fix punctuation, paragraph split)
- Generate notes (Markdown), flashcards (JSON), quizzes (JSON)
- Build FAISS index from cleaned transcript + notes
- RAG chat: retrieve relevant chunks and generate context-aware answers

## Tech Stack
- Python, Streamlit
- Whisper STT (open-source)
- SentenceTransformers embeddings (`all-MiniLM-L6-v2`)
- FAISS vector store
- Optional OpenAI-compatible LLM API; otherwise offline heuristics

## Project Structure
```
ai-learning-companion/
├── app.py                 # Streamlit UI
├── requirements.txt
├── audio/                 # uploaded audio
├── transcripts/
│   ├── raw/
│   └── cleaned/
├── llm/
│   ├── summarizer.py
│   ├── quiz_generator.py
│   ├── flashcards.py
│   └── outputs/
├── rag/
│   ├── embeddings.py
│   ├── vector_store.py
│   └── chat.py
└── utils/
    └── text_cleaner.py
```

## Setup Notes
- You can use the app without any paid services. If `OPENAI_API_KEY` is not set, the app uses deterministic fallbacks for notes/flashcards/quiz and returns context-rich answers in chat.
- For `.mp3` transcription, install FFmpeg or convert to `.wav`. The app handles `.wav` without FFmpeg.

## More Details
See `ai-learning-companion/README.md` for in-depth architecture, workflow diagram, and resume-ready project description.