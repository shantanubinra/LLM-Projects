# 📄 Multimodal RAG Document Assistant

## 🚀 Project Overview

This repository implements a **Multimodal Retrieval‑Augmented Generation (RAG)** system designed to ingest, index, and reason over complex documents. Unlike traditional text‑only RAG pipelines, it extracts and understands both written content and embedded visual data (charts, graphs, images) using vision‑language models.

The platform features a dual‑write vector database architecture for managing both permanent knowledge and isolated temporary sessions, dynamic query routing with conversational memory, exact‑page clickable citations, and a continuous LLM‑as‑a‑judge evaluation loop.

> The application is architected as a collection of decoupled, single‑responsibility modules.



👉 **[Detailed module documentation & screenshots](Module_detials.md)**
<img width="1920" height="1080" alt="Screenshot (114)" src="https://github.com/user-attachments/assets/83c64a12-af49-4756-8cc6-249ecf44c311" />
<img width="1920" height="1080" alt="Screenshot (112)" src="https://github.com/user-attachments/assets/c26ccad7-fed1-4237-8be1-e7fd18032880" />

## 🚀 Project Architecture

flowchart TD
    %% Styling
    classDef ui fill:#4CAF50,stroke:#388E3C,stroke-width:2px,color:white
    classDef processing fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:white
    classDef storage fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:white
    classDef llm fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:white
    classDef observability fill:#607D8B,stroke:#455A64,stroke-width:2px,color:white
    classDef eval fill:#E91E63,stroke:#C2185B,stroke-width:2px,color:white

    subgraph User Interface [Streamlit Frontend]
        A[User Uploads PDF]:::ui
        B[User Asks Question]:::ui
        C[Thumbs Down Feedback]:::ui
    end

    subgraph Ingestion Pipeline [src/parser.py]
        D[Save PDF to static/pdfs/]:::processing
        E[PyMuPDF: Extract Text & Images]:::processing
        F{Vision Enabled?}:::processing
        G[gpt-4o-mini: Summarize Images]:::llm
        H[Chunking & Metadata Tagging]:::processing
        I[text-embedding-3-small]:::llm
    end

    subgraph Vector Database [src/retriever.py]
        J[(ChromaDB: Permanent/Ephemeral)]:::storage
    end

    subgraph RAG & Routing [src/generator.py]
        K[Retrieve Top-K Chunks k=10]:::processing
        L{Score >= 0.7?}:::processing
        M[LLM Rewriter + Chat History]:::llm
        N[gpt-4o: Generate Answer]:::llm
        O[Dynamic HTML Citations]:::processing
    end

    subgraph Observability [Arize Phoenix]
        P([OpenTelemetry Tracing: Logs Routing, Tokens, Latency]):::observability
    end

    subgraph Evaluation Loop [src/evaluator.py & evaluate.py]
        Q[(flagged_responses.jsonl)]:::storage
        R[(test.json)]:::storage
        S[Ragas Evaluator: Faithfulness, Relevancy]:::eval
        T[JSON Evaluation Report]:::eval
    end

    %% Ingestion Flow
    A --> D
    D --> E
    E --> F
    F -- Yes --> G
    G --> H
    F -- No --> H
    H --> I
    I --> J

    %% Interaction Flow
    B --> K
    J <--> K
    K --> L
    L -- No --> M
    M --> K
    L -- Yes --> N
    N --> O
    O --> |Renders Answer & Links| B

    %% Feedback & Eval Flow
    C --> Q
    Q --> S
    R --> S
    S --> T

    %% Observability Connections (Dotted lines)
    G -.-> P
    I -.-> P
    M -.-> P
    N -.-> P

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


