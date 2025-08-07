from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Gzip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Cache for RAG Pipelines
rag_pipelines = {}

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class QuestionResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run")
async def process_questions(request: QuestionRequest):
    try:
        # Use cached RAG Pipeline or create new one
        if request.documents not in rag_pipelines:
            rag_pipelines[request.documents] = RAGPipeline(request.documents)
        
        rag_pipeline = rag_pipelines[request.documents]
        
        # Process all questions concurrently
        answers = []
        for question in request.questions:
            response = await rag_pipeline.aquery(question)
            answers.append(response["result"])
        
        return QuestionResponse(answers=answers)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))