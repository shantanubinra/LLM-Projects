import os
import urllib.parse
import streamlit as st
import phoenix as px
from phoenix.otel import register

# Observability Instrumentors
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from dotenv import load_dotenv
import config

# LangChain message formatting
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# OBSERVABILITY SETUP (MUST BE AT THE TOP)
# ==========================================

# Launch Phoenix background server
if "phoenix_session" not in st.session_state:
    st.session_state.phoenix_session = px.launch_app()

# Register the tracer provider
tracer_provider = register(project_name="manpower-multimodal-rag")

# Explicitly instrument LangChain and OpenAI to capture traces
LangChainInstrumentor().instrument()
OpenAIInstrumentor().instrument()

# ==========================================
# LOCAL MODULE IMPORTS (MUST BE AFTER INSTRUMENTATION)
# ==========================================
from src.parser import MultimodalDocumentParser
from src.retriever import VectorStoreManager
from src.generator import RAGPipeline
from src.evaluator import FeedbackEvaluator
from src.logger import get_logger

# ==========================================
# APP INITIALIZATION
# ==========================================
load_dotenv()
logger = get_logger("streamlit_app")

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("📄 AI Document Assistant")

# Show the Phoenix dashboard link
st.markdown(f"**Phoenix Dashboard:** [Open Traces]({st.session_state.phoenix_session.url})")

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_context" not in st.session_state:
    st.session_state.last_context = ""
if "temp_store" not in st.session_state:
    st.session_state.temp_store = None

# Check for existing permanent database on startup
if "is_indexed" not in st.session_state:
    if os.path.exists(os.path.join("vector_db", "chroma.sqlite3")):
        st.session_state.is_indexed = True
    else:
        st.session_state.is_indexed = False

# Instantiate Core Pipeline Objects
parser = MultimodalDocumentParser()
db_manager = VectorStoreManager()
evaluator = FeedbackEvaluator()

# ==========================================
# SIDEBAR: MODE SELECTION & UPLOADS
# ==========================================
with st.sidebar:
    st.header("⚙️ Chat Mode")
    chat_mode = st.radio(
        "Select Data Source:",
        ["Prebuilt Knowledge Base", "Temporary Uploaded Document"]
    )
    
    st.divider()
    
    st.header("🛠️ Ingestion Settings")
    # THE KILL SWITCH: Defaults to False to save API tokens
    enable_vision = st.checkbox("Extract Charts/Images (Consumes OpenAI Tokens)", value=False)
    st.caption("Enable this only if the PDF relies heavily on visual data.")
    
    st.divider()
    
    if chat_mode == "Prebuilt Knowledge Base":
        st.info("📚 Querying the permanent database.")
        st.markdown("### Add to Knowledge Base")
        uploaded_file = st.file_uploader("Upload PDF to add permanently:", type="pdf", key="kb_upload")
        
        if uploaded_file and st.button("Add to Database"):
            with st.spinner("Extracting and adding to disk..."):
                # Save to the static folder so Streamlit can serve it
                save_dir = os.path.join("static", "pdfs")
                os.makedirs(save_dir, exist_ok=True)
                
                temp_path = os.path.join(save_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Parse with the vision toggle
                docs = parser.parse(temp_path, extract_images=enable_vision)
                db_manager.add_to_store(docs)
                
                st.session_state.is_indexed = True
                st.success("Permanently added to knowledge base!")

    elif chat_mode == "Temporary Uploaded Document":
        st.info("📄 Querying an isolated, temporary document.")
        st.markdown("### Upload Document")
        uploaded_file = st.file_uploader("Upload PDF for temporary chat:", type="pdf", key="temp_upload")
        
        if uploaded_file and st.button("Process Document"):
            with st.spinner("Processing in-memory and archiving to Knowledge Base..."):
                # Save to the static folder so Streamlit can serve it
                save_dir = os.path.join("static", "pdfs")
                os.makedirs(save_dir, exist_ok=True)
                
                # Prefix to avoid overwriting a permanent file with the same name
                file_name = f"temp_{uploaded_file.name}"
                temp_path = os.path.join(save_dir, file_name)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Parse with the vision toggle
                docs = parser.parse(temp_path, extract_images=enable_vision)
                
                # Background Archive
                db_manager.add_to_store(docs)
                st.session_state.is_indexed = True 
                
                # Session Isolation
                st.session_state.temp_store = db_manager.build_temporary_store(docs)
                
                st.success("Ready to chat! (Document also archived to Knowledge Base)")

# ==========================================
# MAIN CHAT INTERFACE
# ==========================================

can_chat = False
active_store = None

if chat_mode == "Prebuilt Knowledge Base" and st.session_state.get("is_indexed"):
    can_chat = True
    active_store = db_manager.get_store() 
elif chat_mode == "Temporary Uploaded Document" and st.session_state.temp_store is not None:
    can_chat = True
    active_store = st.session_state.temp_store

if can_chat:
    # Display previous messages with HTML rendering enabled
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Handle new user input
    if prompt := st.chat_input("Ask about the text or charts..."):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                
                rag_pipeline = RAGPipeline(vector_store=active_store)
                
                message_limit = config.HISTORY_WINDOW * 2 
                formatted_history = []
                
                history_slice = st.session_state.messages[-(message_limit + 1):-1] if len(st.session_state.messages) > 1 else []
                
                for msg in history_slice: 
                    if msg["role"] == "user":
                        formatted_history.append(HumanMessage(content=msg["content"]))
                    else:
                        formatted_history.append(AIMessage(content=msg["content"]))
                
                response = rag_pipeline.generate(prompt, formatted_history)
                
                answer = response["answer"]
                source_docs = response["context"]
                
                st.session_state.last_context = "\n".join([d.page_content for d in source_docs])
                
                # ==========================================
                # CONDITIONAL CITATION GENERATION
                # ==========================================
                # Phrases the LLM uses when the context is missing
                negative_phrases = ["does not contain", "cannot answer", "unable to provide", "not found in the context", "i don't know"]
                
                if any(phrase in answer.lower() for phrase in negative_phrases):
                    # The LLM rejected the context, skip building citations
                    final_answer = answer
                else:
                    citations = set()
                    for doc in source_docs:
                        source_path = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", 1)
                        
                        if source_path != "Unknown":
                            filename = os.path.basename(source_path)
                            encoded_filename = urllib.parse.quote(filename)
                            
                            # HTML anchor tag to force it to open in a new tab
                            link = f'<a href="/app/static/pdfs/{encoded_filename}#page={page}" target="_blank">{filename} (Page {page})</a>'
                            citations.add(link)
                    
                    final_answer = f"{answer}\n\n**Citations:**\n"
                    for citation in citations:
                        final_answer += f"- {citation}\n"
                
                # Render the final output with HTML enabled
                st.markdown(final_answer, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

    # ==========================================
    # CUSTOM FEEDBACK LOOP (LLM-as-a-Judge)
    # ==========================================
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "assistant":
        if st.feedback("thumbs") == 0:
            st.warning("Response flagged! Running LLM-as-a-judge evaluation...")
            
            last_q = st.session_state.messages[-2]["content"]
            last_a = st.session_state.messages[-1]["content"]
            
            evaluator.evaluate_flag(last_q, st.session_state.last_context, last_a)
            st.error("Evaluation completed and logged to flagged_responses.jsonl.")

else:
    if chat_mode == "Prebuilt Knowledge Base":
        st.info("👈 Please add a document to your Knowledge Base to start chatting.")
    else:
        st.info("👈 Please upload a temporary document to start chatting.")