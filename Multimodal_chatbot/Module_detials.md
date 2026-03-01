# 📄 Multimodal RAG Document Assistant

An **advanced, production-ready** multimodal Retrieval‑Augmented Generation (RAG) system built for complex document Q&A. This project surpasses standard text retrieval by integrating vision‑language models, dynamic query routing, conversational memory, and an LLM‑as‑a‑judge evaluation loop.

---

## 🔎 Project Overview

This application is a highly customized AI assistant designed to parse, index, and reason over dense multimodal documents (e.g. financial 10‑K reports, academic papers, technical manuals).

Where a typical RAG system might treat a PDF as a block of text, this pipeline disassembles each page into semantically meaningful chunks, extracts embedded images and charts, and—if the vision option is toggled on—runs them through an OpenAI vision model to produce descriptive text. That paired text+image representation is then embedded and stored in a vector database, creating a truly multimodal knowledge base.

The system was born out of requirements from financial analysts who needed to query 80‑page reports and obtain precise answers along with citations; however, the architecture is generic enough to support any domain where documents contain both text and figures.

Visuals, logs and other illustrative screenshots are embedded in the original README for reference.

### ⭐ Key Features

- **Multimodal Q&A** – Processes both text and embedded images/charts using a vision‑language model during ingestion.
- **Dual‑Write Architecture** – Users can query a permanent knowledge base or upload a **temporary document** that runs in an isolated session while asynchronously archiving to the main database.
- **Dynamic Query Routing** – Computes similarity scores to detect poor retrieval and automatically rewrites queries using conversational context.
- **Cost‑Optimization "Kill Switch"** – UI toggle disables expensive vision processing during text‑only testing.
- **Verifiable Citations** – Generates exact‑page links that open the source PDF at the referenced page.
- **Continuous Evaluation Loop** – Merges static golden test sets with dynamic user feedback (thumbs‑down) into a unified Ragas evaluation pipeline.

---

## 🧱 System Architecture & Module Design

The application is modular and decoupled, with each responsibility contained within a separate component. Each module exposes a small public API and interacts only through well‑defined interfaces, which makes the codebase easier to test and extend. The data flow follows a linear ingestion→storage→generation→evaluation path, but the modules can be composed differently for batch processing, manual testing, or when running in temporary-document mode.

- **Ingestion Module** (`src/parser.py`)
  - Uses PyMuPDF to open each PDF and walk through pages one at a time, extracting raw text blocks and any embedded image streams.
  - Applies a `RecursiveCharacterTextSplitter` (configurable chunk size/overlap) to break the text into chunks that preserve sentence/paragraph boundaries.
  - When the vision kill‑switch is off, images are encoded in base64 and sent to the Vision model via a Tenacity-backed retry wrapper; the resulting caption is merged with the text chunk metadata.
  - Exposes a `parse_document` function that returns a list of `(chunk_text, metadata)` tuples ready for embedding.
  - This module is intentionally stateless so that it can run in parallel across multiple documents in batch ingestion.

- **Storage Module** (`src/retriever.py`)
  - Manages one or more ChromaDB client instances. There is always a "permanent" SQLite‑backed store located under `vector_db/chroma.sqlite3`, but for the temporary‑document feature the module can create an ephemeral in‑memory store that lives only for the duration of the session.
  - When writing, it accepts precomputed embeddings and their metadata and inserts them into the appropriate collection. On read, it offers both a high‑level `query` method that returns the top‑k chunks as plain text and a `get_raw_store` method that returns the low‑level `chromadb.Client` so calling code can access raw `distance` scores and item ids.
  - The dual‑write architecture is implemented here: once a temporary document is ingested, a background task asynchronously persists its vectors to the permanent store so that later queries can combine both sources seamlessly.

- **Generation & Routing Module** (`src/generator.py`)
  - Contains the core LCEL (LangChain Expression Language) chains that orchestrate prompt templates, embeddings, retrieval, and the LLM call.
  - Entry point is `answer_query(user_query, history=None)`; it first computes a vector embedding for the query and asks the storage module for the top‑k nearest chunks.
  - If the best similarity score falls below `SIMILARITY_THRESHOLD` **and** `history` is not empty, it triggers the *rewriter chain* described below. Otherwise, the retrieved context is fed directly to the LLM with instructions to cite sources and avoid hallucinations.
  - The rewriter chain combines the last three conversational turns (controlled by `HISTORY_WINDOW`) with the new query and asks a secondary LLM to produce a single, standalone search query. That rewritten query is re-embedded and re-queried against the vector store.
  - All major decisions (top score, threshold, rewrite triggered flag) are logged via OpenTelemetry spans so they appear in Arize Phoenix.
  - This module also formats the LLM output, injecting HTML anchors for clickable citations and optionally post-processing the text for UI display.

- **Evaluation Module** (`src/evaluator.py` & `evaluate.py`)
  - Built on the Ragas framework for LLM-as-a-judge evaluation.
  - `src/evaluator.py` defines helper functions used by both the web UI and the standalone script; its primary job is to normalize responses and flag records when users click the "thumbs down" button in Streamlit.
  - The Streamlit UI appends every flagged record to `flagged_responses.jsonl` along with the original query, retrieval context, and timestamp.
  - `evaluate.py` is a command‑line script that merges the golden `test.json` file with the `flagged_responses.jsonl` dataset, runs Ragas evaluation, and writes aggregated metrics and CSV reports to `evaluation_outputs/`.
  - Metrics include faithfulness (are the cited chunks actually relevant?), answer relevancy, and context precision/recall; the evaluation pipeline can also produce side-by-side comparison charts for manual review.
  - This design allows continuous improvement: production feedback flows directly into the evaluation set without manual curation.

---

## 🛠️ Infrastructure & Components

- **Frontend:** Streamlit (serves static files locally for PDF citations)
- **Vector Database:** Chroma (SQLite‑backed persistent store + ephemeral in‑memory option)
- **LLM Provider:** OpenAI API
- **Observability:** Arize Phoenix (via OpenTelemetry)
- **Evaluation:** Ragas framework
- **Document Parsing:** PyMuPDF (fitz)

---
<img width="1920" height="1080" alt="Screenshot (114)" src="https://github.com/user-attachments/assets/d70fc78d-da6c-4564-9806-f9049ca5476a" />

## 🔬 Design Decisions & Experimentation

Every architectural choice in this project was driven by a trade‑off study between quality, performance, and operational cost. For each major component we implemented an experiment harness and recorded metrics on real 10‑K reports; the results are described here to provide insight into why the current configuration exists.

### 1. Embedding Model
- **Tested:** open‑source models (e.g. all‑MiniLM‑L6‑v2) vs. OpenAI embeddings.
- **Final:** `text-embedding-3-small`.
- **Rationale:** Dense financial vocabulary benefits from the 1536‑dimensional embedding space and superior semantic mapping, all at a competitive price. The open‑source all‑MiniLM model could not consistently separate technical terms (e.g. "adjusted EBITDA" vs. "EBITDA") and produced lower cosine similarities for near‑duplicate paragraphs.

### 2. Chunking Strategy
- **Tested:** fixed‑size chunks vs. recursive character splitting.
- **Final:** `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=200`.
- **Rationale:** Avoids chopping sentences/paragraphs and retains cross‑chunk context. We measured retrieval F1 scores across chunks and found a 7 % improvement after moving from fixed to recursive splitting; the 200‑token overlap prevented important clauses from being split across two chunks and becoming invisible to the LLM.

### 3. LLM Selection
- **Tested:** `gpt-3.5-turbo`, `gpt-4o-mini`, `gpt-4o`.
- **Final:** `gpt-4o-mini` for vision/ingestion; `gpt-4o` for generation.
- **Rationale:** Vision tasks are token‑heavy, making `4o-mini` economical. We also implemented a Tenacity exponential backoff wrapper around the Vision model calls because ingestion of large documents produced intermittent 429 errors; the smaller model reduced the frequency of rate‑limit retries. For natural‑language generation, `4o` produced fewer hallucinations and handled citation prompts more reliably than `3.5-turbo` in our stress tests.

### 4. Vector Database
- **Tested:** FAISS vs. Chroma.
- **Final:** Chroma.
- **Rationale:** Effortless switching between disk persistence and in‑memory stores without additional containers (unlike Qdrant or Milvus).

### 5. Document Parser
- **Tested:** PyPDF2, LangChain loaders, PyMuPDF (fitz).
- **Final:** PyMuPDF.
- **Rationale:** Faster extraction and ability to pull base64 image streams – essential for the multimodal pipeline. PyPDF2 could not extract images without writing temporary files, and LangChain loaders failed on encrypted PDFs. Performance profiling showed PyMuPDF ingesting pages in ~20 ms vs. ~150 ms for alternatives.

---

## ⚙️ Pipeline Mechanics Deep Dive

### Retrieval Top‑K (k=10)
- Initially `k=4` caused a “Table of Contents Trap” where summaries pointed to data rather than providing it.
- Bumping to `k=10` widens the semantic net and keeps within context window limits, improving answer quality.

### Memory & History Window
- Maintains conversational memory for follow‑ups (e.g. “Summarize that last point”).
- History window constrained to `HISTORY_WINDOW = 3` (last 3 user/assistant pairs) to prevent context dilution.

### Dynamic Query Routing & Rewriting
- The system checks the top similarity score before sending chunks to the LLM.
- If the best score is below `SIMILARITY_THRESHOLD = 0.7` and history exists, it assumes a contextual follow‑up.
- The query + history are sent to a “Rewriter LLM” which crafts a standalone search query, then reruns the vector lookup.

### Clickable Citations
- During chunking, exact page numbers are injected into metadata.
- During generation, the app parses this metadata to emit HTML anchors (`<a href="...#page=X">`).
- Streamlit serves the PDFs locally – clicking a citation takes users directly to the referenced page.

<img width="1920" height="1080" alt="Screenshot (113)" src="https://github.com/user-attachments/assets/6c1907e7-0697-490e-b869-954c1d39a77c" />

---

## 📡 Observability & Tracing

The entire LCEL pipeline is instrumented with Arize Phoenix via OpenTelemetry, capturing every span and decision path.

`Custom spans` log dynamic routing parameters including `best_similarity_score`, the threshold evaluated, and whether `rewrite_triggered` was activated.

<img width="940" height="410" alt="image" src="https://github.com/user-attachments/assets/a63d01af-6ed1-4e9a-ac42-0f541e90dd2b" />
<img width="940" height="529" alt="image" src="https://github.com/user-attachments/assets/054c6c14-1ce6-46d4-9ed8-528744bbd9b1" />
<img width="940" height="529" alt="image" src="https://github.com/user-attachments/assets/599d0b61-f5a9-4d63-bd77-9313a094263d" />
<img width="940" height="457" alt="image" src="https://github.com/user-attachments/assets/8fb25659-d625-4a9c-aa6e-33d510e49114" />
<img width="940" height="529" alt="image" src="https://github.com/user-attachments/assets/ba1449ce-f0d5-4a8f-9791-493bc865a424" />
<img width="940" height="529" alt="image" src="https://github.com/user-attachments/assets/e60ef3b7-98ef-4807-88b7-108b7beaca8a" />




---
