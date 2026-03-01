📄 Multimodal RAG Document Assistant
🚀 Project Overview
This project is a Multimodal Retrieval-Augmented Generation (RAG) system built to ingest, index, and reason over complex documents. Unlike standard text-only RAG pipelines, this system is capable of extracting and understanding both text and embedded visual data (charts, graphs, images) using vision-language models.

It features a dual-write vector database architecture for managing both permanent knowledge and isolated temporary sessions, dynamic query routing with conversational memory, exact-page clickable citations, and a continuous LLM-as-a-judge evaluation loop.

This application is highly decoupled into single-responsibility modules. 

👉 **[Click here to read the Detailed Module Documentation & View Output Screenshots](Module_detials.md)**

🛠️ Key Components & Infrastructure
Frontend/UI: Streamlit (with local static file serving for PDF rendering)

Orchestration: LangChain (using LCEL syntax)

LLMs: OpenAI gpt-4o (Reasoning & Generation), gpt-4o-mini (Vision/Ingestion)

Embeddings: OpenAI text-embedding-3-small

Vector Database: ChromaDB (Local SQLite-backed & Ephemeral In-Memory)

Document Parsing: PyMuPDF (fitz) for text and base64 image extraction

Observability: Arize Phoenix (via OpenTelemetry)

Evaluation: Ragas Framework

🧩 Module Design & Architecture
The application is highly decoupled into single-responsibility modules:

app.py (The Orchestrator): The main Streamlit application. Manages session states, UI rendering, database toggling (Permanent vs. Temporary), static file routing for citations, and user feedback collection (thumbs down/flagging).

src/parser.py (The Ingestion Engine): Uses PyMuPDF to parse documents page-by-page. It includes an optional, Tenacity-backed (exponential backoff) Vision API integration to summarize charts and images into text chunks.

src/retriever.py (The Storage Manager): Handles all ChromaDB interactions. It exposes methods to write to the permanent disk-backed database or spin up ephemeral, in-memory vector stores for isolated document chats.

src/generator.py (The Reasoning Engine): Contains the core LangChain logic. Implements Dynamic Query Routing: if the initial mathematical retrieval score is too low, it autonomously rewrites the user's query using conversational history before fetching context. All routing decisions are logged via custom OpenTelemetry spans.

src/evaluator.py & evaluate.py (The Evaluation Loop): * evaluator.py captures flagged responses from the UI and saves them to a .jsonl file.

evaluate.py is a standalone script that uses Ragas to evaluate both a golden test set (test.json) and the user-flagged responses, outputting a unified metrics report (Faithfulness, Answer Relevancy, Context Precision/Recall).


project_root/
│
├── .streamlit/
│   └── config.toml                # Enables static file serving for PDF citations
├── data/                          # Temporary storage for uploaded raw PDFs
├── static/
│   └── pdfs/                      # Static serving directory for clickable citations
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
├── config.py                      # Global thresholds and configurations
├── evaluate.py                    # Standalone Ragas evaluation script
├── test.json                      # Golden test set for evaluations
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation


⚙️ Project Setup & Installation Guide
Follow these steps to configure the environment and run the application locally.

1. Prerequisites
Ensure you have Python 3.9 or higher installed on your system.

2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies. Open your terminal and run:

**For Windows (Command Prompt / PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS / Linux / WSL:**
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install Dependencies
With your virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```
4. Configure Environment Variables
Create a .env file in the root directory of the project and add your OpenAI API key:

```bash
OPENAI_API_KEY="your-openai-api-key-here"
```
5. Configure Streamlit for Static File Serving
For the clickable PDF citations to work properly, Streamlit must be allowed to serve local files.
Create a .streamlit folder in your root directory, create a config.toml file inside it, and add the following:

```toml
[server]
enableStaticServing = true
```
6. Run the Application
Start the Streamlit server:

```bash
streamlit run app.py
```

The app will open automatically in your browser at http://localhost:8501.
The Arize Phoenix observability dashboard will launch automatically in the background.

7. Run the Evaluation Pipeline (Optional)
To run the Ragas LLM-as-a-judge evaluation on the test set and any user-flagged responses:

```bash
python evaluate.py
```