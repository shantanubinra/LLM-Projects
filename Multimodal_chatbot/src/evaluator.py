import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.logger import get_logger

logger = get_logger(__name__)

class FeedbackEvaluator:
    def __init__(self, model_name: str = "gpt-4o-mini", log_file: str = "flagged_responses.jsonl"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.log_file = log_file
        self.prompt = PromptTemplate.from_template("""
            You are an impartial evaluator. Assess the following response based on the provided context.
            Rate each metric from 1 to 5 (5 being best) and provide a brief reason.
            
            Question: {question}
            Context: {context}
            Answer: {answer}
            
            Metrics:
            1. Relevance: Does the answer directly address the question?
            2. Completeness: Does the answer contain all necessary details from the context?
            3. Consistency: Is the answer free of hallucinations?
            
            Output strictly in JSON format: {{"relevance": int, "completeness": int, "consistency": int, "reasoning": "string"}}
        """)
        self.chain = self.prompt | self.llm

    def evaluate_flag(self, question: str, context: str, answer: str):
        logger.info("Initiating LLM-as-a-Judge evaluation for flagged response.")
        result = self.chain.invoke({"question": question, "context": context, "answer": answer})
        
        flagged_data = {"question": question, "answer": answer, "evaluation": result.content}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(flagged_data) + "\n")
            
        logger.warning(f"Flagged response logged: {result.content}")