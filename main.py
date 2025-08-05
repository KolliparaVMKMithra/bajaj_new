from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List
from app.rag_pipeline import RAGPipeline
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Insurance Policy Q&A API")

# Initialize RAG Pipeline
pdf_path = "data/Arogya_Sanjeevani_Policy.pdf"
rag_pipeline = RAGPipeline(pdf_path)

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class QuestionResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run")
async def process_questions(
    request: QuestionRequest,
    authorization: str = Header(..., description="Bearer <api_key>")
):
    # Verify API key
    api_key = authorization.replace("Bearer ", "")
    if api_key != os.getenv("SECURITY_API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Process each question
        answers = []
        for question in request.questions:
            response = rag_pipeline.query(question)
            answers.append(response["result"])
        
        return QuestionResponse(answers=answers)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")