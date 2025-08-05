import os
from typing import List, Dict, Any
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, pdf_path: str):
        logger.info("Initializing Enhanced RAG Pipeline...")
        self.pdf_path = pdf_path
        self.documents = self._load_documents()
        self.llm = self._create_llm()
        self.embeddings = self._create_embeddings()
        self.vector_store = self._create_vector_store()
        self.prompt = self._create_prompt()
        logger.info("RAG Pipeline initialization complete.")

    def _load_documents(self) -> List[Any]:
        """Enhanced document loading with precise chunking."""
        logger.info(f"Loading document: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", ".", "!", "?", ",", " ", ""],
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "source": self.pdf_path,
                "chunk_size": len(chunk.page_content)
            })
        
        logger.info(f"Created {len(chunks)} document chunks")
        return chunks

    def _create_llm(self) -> AzureChatOpenAI:
        """Initialize LLM with optimal parameters."""
        return AzureChatOpenAI(
            openai_api_version=os.environ["OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            temperature=0.0,
            max_tokens=500,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

    def _create_embeddings(self) -> AzureOpenAIEmbeddings:
        """Initialize embeddings model."""
        return AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["OPENAI_API_VERSION"]
        )

    def _create_vector_store(self) -> FAISS:
        """Create FAISS vector store."""
        return FAISS.from_documents(self.documents, self.embeddings)

    def _create_prompt(self) -> PromptTemplate:
        """Create enhanced prompt template."""
        template = """
        You are an expert insurance policy analyst for the National Parivar Mediclaim Plus Policy. 
        Analyze the provided context and answer the question with high accuracy.

        Important Guidelines:
        1. ONLY provide information explicitly stated in the policy document
        2. Include specific numbers, limits, conditions, and waiting periods
        3. Format monetary values as "Rs. X"
        4. For unavailable information, state "This information is not explicitly mentioned in the policy document"
        5. For coverage questions, mention all applicable sub-limits and conditions
        6. For waiting periods, specify exact durations and exceptions

        Policy Context:
        {context}

        Question: {question}

        Step-by-step analysis:
        1. Identify relevant policy sections
        2. Extract exact terms and conditions
        3. Present information clearly and precisely

        Detailed Answer:"""
        
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def _validate_response(self, question: str, answer: str, context: str) -> str:
        """Validate response accuracy."""
        validation_prompt = f"""
        Verify the accuracy of this answer against the policy context:
        Question: {question}
        Given Answer: {answer}
        Context: {context}

        Verification Steps:
        1. Check factual accuracy
        2. Verify completeness
        3. Confirm relevance
        4. Ensure all numbers and conditions are correct

        Provide the corrected answer if needed:"""

        return self.llm.predict(validation_prompt)

    def _calculate_confidence(self, question: str, answer: str) -> float:
        """Calculate confidence score."""
        try:
            question_embedding = self.embeddings.embed_query(question)
            answer_embedding = self.embeddings.embed_query(answer)
            similarity = sum(q * a for q, a in zip(question_embedding, answer_embedding))
            return round(max(0.0, min(1.0, similarity)), 2)
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0

    def query(self, question: str) -> Dict[str, Any]:
        """Process query with enhanced accuracy."""
        logger.info(f"Processing question: {question}")
        try:
            # Get relevant documents
            relevant_docs = self.vector_store.similarity_search(question, k=5)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Generate initial response
            chain = LLMChain(llm=self.llm, prompt=self.prompt)
            initial_response = chain.invoke({
                "context": context,
                "question": question
            })
            
            # Validate response
            validated_response = self._validate_response(
                question=question,
                answer=initial_response["text"],
                context=context
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(question, validated_response)
            
            response = {
                "result": validated_response.strip(),
                "confidence_score": confidence,
                "sources": [
                    {
                        "page": doc.metadata["page"],
                        "content": doc.page_content[:200]
                    } 
                    for doc in relevant_docs[:2]
                ]
            }
            
            logger.info(f"Query processed successfully. Confidence: {confidence}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "result": "Error processing query. Please try again.",
                "confidence_score": 0.0,
                "sources": []
            }