import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
TOP_K = 5
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"
EVAL_LLM_MODEL = "gpt-4o-mini"
VECTOR_DB_PATH = "./vector_db/index.faiss"

SIMILARITY_THRESHOLD = 0.7


HISTORY_WINDOW = 3

CHUNK_K = 10

EVAL_DATA_PATH = os.getenv("EVAL_DATA_PATH")
FEEDBACK_OUTPUT_PATH = os.getenv("FEEDBACK_OUTPUT_PATH")
EVALUATION_OUTPUT_PATH = os.getenv("EVALUATION_OUTPUT_PATH")
