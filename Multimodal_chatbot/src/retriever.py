from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.logger import get_logger

logger = get_logger(__name__)

class VectorStoreManager:
    def __init__(self, db_dir: str = "./vector_db", embedding_model: str = "text-embedding-3-small"):
        self.db_dir = db_dir
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
    def add_to_store(self, documents: List[Document], chunk_size: int = 1000, overlap: int = 200):
        """Permanently appends documents to the local disk vector database."""
        logger.info("Adding to the permanent vector store.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = splitter.split_documents(documents)
        
        vector_db = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
        vector_db.add_documents(chunks)

    def get_store(self):
        """Returns the raw Chroma vector store so we can access similarity scores directly."""
        return Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)

    def build_temporary_store(self, documents: List[Document], chunk_size: int = 1000, overlap: int = 200):
        """Builds an ephemeral, in-memory vector store for a single session."""
        logger.info("Building temporary in-memory vector store.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = splitter.split_documents(documents)
        
        return Chroma.from_documents(chunks, self.embeddings)