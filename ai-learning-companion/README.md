# LectureMind AI — AI-powered Learning Companion

LectureMind AI is an end-to-end, local-first web application that turns lecture audio into structured study materials and enables Retrieval-Augmented Generation (RAG) chat with your lectures.

## Project Overview
- Upload `.mp3`/`.wav` audio
- Transcribe to text using Whisper (open-source)
- Clean transcript (remove fillers, fix punctuation, split paragraphs)
- Auto-generate notes (Markdown), flashcards (JSON), and quizzes (JSON)
- Build a FAISS vector index from cleaned transcript + notes
- Ask questions with RAG — retrieve relevant chunks and answer contextually

## Tech Stack
- Language: Python
- UI: Streamlit
- Speech-to-text: Whisper (open-source)
- LLM Integration: OpenAI-compatible API (optional) OR heuristic fallback (no paid service)
- RAG Framework: Minimal custom pipeline with LangChain-inspired chunking and FAISS
- Vector Database: FAISS (local)
- Embeddings: SentenceTransformers (`all-MiniLM-L6-v2`)
- Storage: Local filesystem

## Architecture
- `app.py`: Streamlit single-page dashboard orchestrating the pipeline
- `utils/text_cleaner.py`: transcript cleaning utilities
- `llm/summarizer.py`: notes generation via OpenAI-compatible API or heuristic fallback
- `llm/flashcards.py`: flashcards generator (OpenAI or heuristic)
- `llm/quiz_generator.py`: quiz generator (OpenAI or heuristic)
- `rag/embeddings.py`: SentenceTransformers embedding wrapper
- `rag/vector_store.py`: FAISS-based per-lecture vector store
- `rag/chat.py`: RAG QA — retrieve chunks and answer via LLM or fallback

### Storage Layout
```
ai-learning-companion/
├── audio/                       # uploaded audio files
├── transcripts/
│   ├── raw/                     # raw transcripts from Whisper
│   └── cleaned/                 # cleaned transcripts
├── llm/
│   ├── summarizer.py
│   ├── quiz_generator.py
│   ├── flashcards.py
│   └── outputs/                 # generated notes/flashcards/quiz
├── rag/
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── chat.py
│   └── faiss_index/             # per-lecture FAISS index + texts.json
├── utils/
│   └── text_cleaner.py
├── app.py
├── requirements.txt
└── README.md
```

## Workflow Diagram (Text-based)
```
[Upload Audio] -> [Whisper STT] -> [Raw Transcript]
                                 -> [Cleaning] -> [Cleaned Transcript]
                                                 -> [Content Generation]
                                                    -> notes.md
                                                    -> flashcards.json
                                                    -> quiz.json
                                                 -> [Chunk & Embed]
                                                    -> FAISS Vector Store
                                 -> [RAG Chat]
                                    -> Retrieve relevant chunks
                                    -> Generate answer via LLM or fallback
```

## Setup Instructions
1. Ensure you are in the project directory and using a Python virtual environment.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. (Optional) Set OpenAI-compatible variables to use an LLM:
   - `OPENAI_API_KEY` (required if using OpenAI or compatible provider)
   - `OPENAI_API_BASE` (optional; defaults to `https://api.openai.com/v1`)
   - `OPENAI_MODEL` (optional; defaults to `gpt-3.5-turbo`)
4. Run the app:
   ```
   streamlit run app.py
   ```

### Notes on Local-First Operation
- You do NOT need an API key to use the basic version. When no `OPENAI_API_KEY` is set, the app uses heuristic fallbacks for notes, flashcards, and quizzes and returns context-rich answers in chat.
- Whisper and SentenceTransformers will download models on first use.

## How RAG Works Here
- Cleaned transcript and generated notes are split into chunks.
- Chunks are embedded using SentenceTransformers (`all-MiniLM-L6-v2`).
- Embeddings are stored in a per-lecture FAISS index with `IndexFlatIP` (cosine on normalized vectors).
- During chat, the user’s question is embedded, the top-k chunks are retrieved, and the answer is generated using either an OpenAI-compatible model or a heuristic fallback that cites the retrieved context.

## Resume-Ready Project Description
LectureMind AI is a full-stack AI application built with Python and Streamlit, featuring a local-first pipeline that transforms lecture audio into structured study materials. It integrates Whisper for speech-to-text, custom text cleaning, and AI content generation (notes, flashcards, quizzes) with an optional OpenAI-compatible LLM. The app implements a RAG workflow using SentenceTransformers for embeddings and FAISS for vector search, enabling students to chat with their lecture content and receive context-aware answers. The codebase follows modular design, includes error handling, and is suitable for internship demos and GitHub submission.

## Troubleshooting
- Whisper requires `torch`; ensure `torch` is installed per your environment (CPU/GPU).
- FAISS on Windows: install `faiss-cpu` (provided in `requirements.txt`).
- If you see model download failures, check internet connectivity and retry.