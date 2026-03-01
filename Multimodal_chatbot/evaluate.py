import os
import json
import pandas as pd
from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from dotenv import load_dotenv

# Import your pipeline components
from src.retriever import VectorStoreManager
from src.generator import RAGPipeline
from src.logger import get_logger
import config
logger = get_logger("evaluation_pipeline")
load_dotenv()

def run_evaluation():
    logger.info("Starting combined evaluation pipeline...")
    
    # 1. Setup Output Directory & Filename for JSON
    os.makedirs("evaluation_outputs", exist_ok=True)
    today_date = datetime.now().strftime("%Y-%m-%d")
    output_file = f"{config.EVALUATION_OUTPUT_PATH}_evaluation_result_{today_date}.json"

    # Initialize RAG Pipeline to generate answers for the test.json questions
    db_manager = VectorStoreManager()
    vector_store = db_manager.get_store()
    rag_pipeline = RAGPipeline(vector_store=vector_store)

    test_df = pd.DataFrame()
    flagged_df = pd.DataFrame()

    # ==========================================
    # 2. PROCESS TEST.JSON (The Golden Dataset)
    # ==========================================
    if os.path.exists(config.EVAL_DATA_PATH):
        logger.info("Processing test.json...")
        with open(config.EVAL_DATA_PATH, "r") as f:
            test_items = json.load(f)
            
        test_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
        
        for item in test_items:
            q = item["question"]
            gt = item["ground_truth"]
            
            # Run the query through your pipeline to get the LLM's answer and retrieved chunks
            response = rag_pipeline.generate(q, chat_history=[])
            
            test_data["question"].append(q)
            test_data["answer"].append(response["answer"])
            # Ragas expects contexts as a list of strings
            test_data["contexts"].append([doc.page_content for doc in response["context"]])
            test_data["ground_truth"].append(gt)
            
        test_dataset = Dataset.from_dict(test_data)
        
        logger.info("Running Ragas metrics on test set (Faithfulness, Relevancy, Precision, Recall)...")
        test_results = evaluate(
            test_dataset, 
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        test_df = test_results.to_pandas()
        test_df["data_source"] = "test.json" 
        
    # ==========================================
    # 3. PROCESS FLAGGED RESPONSES (User Feedback)
    # ==========================================
    if os.path.exists("flagged_responses.jsonl"):
        logger.info("Processing flagged_responses.jsonl...")
        flagged_data = {"question": [], "answer": [], "contexts": []}
        
        with open(config.FEEDBACK_OUTPUT_PATH, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    flagged_data["question"].append(item.get("question", ""))
                    flagged_data["answer"].append(item.get("answer", ""))
                    
                    # Ensure context is wrapped in a list for Ragas
                    ctx = item.get("context", "")
                    flagged_data["contexts"].append([ctx] if isinstance(ctx, str) else ctx)

        if flagged_data["question"]:
            flagged_dataset = Dataset.from_dict(flagged_data)
            
            logger.info("Running Ragas metrics on flagged set (Faithfulness, Relevancy only)...")
            flagged_results = evaluate(
                flagged_dataset, 
                metrics=[faithfulness, answer_relevancy]
            )
            flagged_df = flagged_results.to_pandas()
            flagged_df["data_source"] = "flagged_responses.jsonl"

    # ==========================================
    # 4. MERGE AND EXPORT TO JSON
    # ==========================================
    if not test_df.empty or not flagged_df.empty:
        final_report = pd.concat([test_df, flagged_df], ignore_index=True)
        
        # Save to JSON format
        final_report.to_json(output_file, orient="records", indent=4)
        logger.info(f"✅ Full evaluation complete! Results saved to {output_file}")
    else:
        logger.warning("No data found in test.json or flagged_responses.jsonl to evaluate.")

if __name__ == "__main__":
    run_evaluation()