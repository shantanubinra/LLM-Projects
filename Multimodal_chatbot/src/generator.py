import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from opentelemetry import trace
from src.logger import get_logger

logger = get_logger(__name__)

# Initialize the OpenTelemetry tracer for this module
tracer = trace.get_tracer(__name__)

class RAGPipeline:
    def __init__(self, vector_store, model_name: str = "gpt-4o"):
        self.vector_store = vector_store  # <-- This is the exact parameter app.py is looking for!
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # 1. Prompt for standard Q&A
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert data assistant. Answer the user's question using ONLY the context provided below.\n\nContext: {context}"),
            ("human", "{input}")
        ])
        self.qa_chain = self.qa_prompt | self.llm
        
        # 2. Prompt for rewriting the query using history
        self.rewrite_prompt = ChatPromptTemplate.from_template(
            "Given the following conversation history and a follow up question, "
            "rephrase the follow up question to be a highly specific, standalone search query.\n\n"
            "Chat History:\n{history}\n\n"
            "Follow Up Input: {input}\n"
            "Standalone query:"
        )
        self.rewrite_chain = self.rewrite_prompt | self.llm

    def generate(self, user_input: str, chat_history: list) -> dict:
        
        # Wrap the custom routing logic in a Phoenix-compatible OpenTelemetry Span
        with tracer.start_as_current_span("Dynamic_Query_Routing") as span:
            
            logger.info(f"Attempting initial retrieval for: {user_input}")
            docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(user_input, k=config.CHUNK_K)
            
            # Extract top score safely
            best_score = docs_and_scores[0][1] if docs_and_scores else 0.0
            
            # Log the scores and thresholds into the Phoenix Dashboard
            span.set_attribute("routing.initial_query", user_input)
            span.set_attribute("routing.best_similarity_score", float(best_score))
            span.set_attribute("routing.similarity_threshold", float(config.SIMILARITY_THRESHOLD))
            
            final_query = user_input
            rewrite_triggered = False
            
            # Conditional Check against config.py threshold
            if best_score < config.SIMILARITY_THRESHOLD and chat_history:
                rewrite_triggered = True
                logger.warning(f"Score below threshold ({config.SIMILARITY_THRESHOLD}). Triggering query rewrite loop.")
                
                # Format history into a readable string
                history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
                
                # Rewrite the query
                rewritten_response = self.rewrite_chain.invoke({
                    "history": history_text, 
                    "input": user_input
                })
                final_query = rewritten_response.content
                logger.info(f"Rewritten query generated: {final_query}")
                
                # Log the rewritten query to Phoenix
                span.set_attribute("routing.rewritten_query", final_query)
                
                # Retrieve again with the newly enriched query
                docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(final_query, k=4)

            span.set_attribute("routing.rewrite_triggered", rewrite_triggered)
            
            # Extract just the documents from the tuple
            context_docs = [doc for doc, score in docs_and_scores]
            context_text = "\n\n".join([doc.page_content for doc in context_docs])
            
        # Generate final answer
        response = self.qa_chain.invoke({
            "context": context_text, 
            "input": final_query 
        })
        
        return {
            "answer": response.content,
            "context": context_docs,
        }