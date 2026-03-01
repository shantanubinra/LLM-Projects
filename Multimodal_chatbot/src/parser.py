import fitz
import base64
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.logger import get_logger

logger = get_logger(__name__)

class MultimodalDocumentParser:
    def __init__(self, vision_model: str = "gpt-4o-mini", max_tokens: int = 250):
        self.llm = ChatOpenAI(model=vision_model, max_tokens=max_tokens)
    
    def _encode_image(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode('utf-8')
        
    def _summarize_image(self, base64_img: str) -> str:
        logger.info("Generating summary for extracted image...")
        msg = self.llm.invoke([
            HumanMessage(content=[
                {"type": "text", "text": "Describe this image, chart, or table in detail. Extract all data points."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ])
        ])
        return msg.content

    def parse(self, file_path: str, extract_images: bool = False) -> List[Document]:
        logger.info(f"Starting extraction for {file_path}. Vision enabled: {extract_images}")
        doc = fitz.open(file_path)
        documents = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            
            # Only trigger the expensive OpenAI Vision calls if the user explicitly opts in
            if extract_images:
                for img_index, img in enumerate(page.get_images()):
                    base_image = doc.extract_image(img[0])
                    b64_img = self._encode_image(base_image["image"])
                    
                    image_summary = self._summarize_image(b64_img)
                    text += f"\n\n[Chart/Image Description]: {image_summary}"
                
            if text.strip():
                metadata = {"source": file_path, "page": page_num + 1}
                documents.append(Document(page_content=text, metadata=metadata))
                
        return documents