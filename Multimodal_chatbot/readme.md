# 📄 Multimodal RAG Document Assistant

## 🚀 Project Overview

This repository implements a **Multimodal Retrieval‑Augmented Generation (RAG)** system designed to ingest, index, and reason over complex documents. Unlike traditional text‑only RAG pipelines, it extracts and understands both written content and embedded visual data (charts, graphs, images) using vision‑language models.

The platform features a dual‑write vector database architecture for managing both permanent knowledge and isolated temporary sessions, dynamic query routing with conversational memory, exact‑page clickable citations, and a continuous LLM‑as‑a‑judge evaluation loop.

> The application is architected as a collection of decoupled, single‑responsibility modules.

## Architecture detials

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/0f46a19b-f236-43ca-a33a-6cb28f242363" />


👉 **[Detailed module documentation & screenshots](Module_detials.md)**
<img width="1920" height="1080" alt="Screenshot (114)" src="https://github.com/user-attachments/assets/83c64a12-af49-4756-8cc6-249ecf44c311" />
<img width="1920" height="1080" alt="Screenshot (112)" src="https://github.com/user-attachments/assets/c26ccad7-fed1-4237-8be1-e7fd18032880" />


## 🛠️ Key Components & Infrastructure

- **Frontend/UI:** Streamlit (serves static files locally for PDF rendering)
- **Orchestration:** LangChain (LCEL syntax)
- **LLMs:** OpenAI `gpt-4o` (reasoning/generation), `gpt-4o-mini` (vision/ingestion)
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Vector Database:** ChromaDB (persistent SQLite + ephemeral in‑memory)
- **Document Parsing:** PyMuPDF (`fitz`) for text and base64 image extraction
- **Observability:** Arize Phoenix (via OpenTelemetry)
- **Evaluation:** Ragas framework

---

## 🧩 Module Design & Architecture

Each component handles a focused concern:

- **app.py** *(Orchestrator)* – main Streamlit app. Manages session state, UI, database toggling (permanent vs temporary), static file routing for citations, and user feedback.
- **src/parser.py** *(Ingestion Engine)* – parses documents page by page with PyMuPDF; optionally calls the Vision API (with Tenacity‑backed retries) to summarize charts/images.
- **src/retriever.py** *(Storage Manager)* – handles ChromaDB interactions; supports both disk‑backed and ephemeral in‑memory stores.
- **src/generator.py** *(Reasoning Engine)* – core LangChain logic; implements dynamic query routing with conversational history and logs decisions via custom OpenTelemetry spans.
- **src/evaluator.py & evaluate.py** *(Evaluation Loop)* – `evaluator.py` captures flagged UI responses to `.jsonl`; `evaluate.py` runs Ragas against a golden `test.json` dataset plus flagged responses, producing metrics (faithfulness, relevancy, context precision/recall).

---

## 📁 Project Layout

```
project_root/
│
├── .streamlit/
│   └── config.toml                # Static file serving for PDF citations
├── data/                          # Temporary storage for uploaded raw PDFs
├── static/
│   └── pdfs/                      # Static directory used for citations
├── vector_db/                     # Persistent Chroma SQLite database
├── evaluation_outputs/            # Generated Ragas JSON/CSV reports
│
├── src/
│   ├── __init__.py
│   ├── evaluator.py
│   ├── generator.py
│   ├── logger.py
│   ├── parser.py
│   └── retriever.py
│
├── app.py                         # Main Streamlit application
├── config.py                      # Global thresholds/config
├── evaluate.py                    # Standalone Ragas evaluation script
├── test.json                      # Golden evaluation test set
├── requirements.txt               # Dependencies
└── README.md                      # This documentation
```

---

## ⚙️ Setup & Installation Guide

Follow these steps to configure and run the application locally.

1. **Prerequisites** – Python 3.9+ is required.

2. **Create a virtual environment** (recommended):
   - **Windows (CMD/PowerShell):**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **macOS/Linux/WSL:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the project root containing:
   ```bash
   OPENAI_API_KEY="your-openai-api-key-here"
   ```

5. **Enable Streamlit static serving:**
   Ensure `.streamlit/config.toml` exists with:
   ```toml
   [server]
   enableStaticServing = true
   ```

6. **Run the application:**
   ```bash
   streamlit run app.py
   ```
   The app will open at `http://localhost:8501` and Arize Phoenix will start in the background.

7. **(Optional) Evaluate with Ragas:**
   ```bash
   python evaluate.py
   ```
   This executes the Ragas LLM‑as‑a‑judge evaluation on the golden test set and any flagged responses.

---


