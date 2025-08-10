# Insurance Policy Q&A API

A sophisticated Question-Answering system built for insurance policy documents using Azure OpenAI and FastAPI. This system uses RAG (Retrieval-Augmented Generation) to provide accurate answers from policy documents.

## 🚀 Features

- **Intelligent Document Processing**: Automatically processes PDF insurance policy documents
- **Advanced RAG Pipeline**: Uses state-of-the-art retrieval augmented generation
- **High Accuracy**: Optimized chunk sizes and context retrieval for precise answers
- **Async Support**: Built with async processing for better performance
- **API Security**: Includes API key authentication
- **Production Ready**: Docker support and Azure deployment configurations

## 🛠️ Tech Stack

- **Backend Framework**: FastAPI
- **AI/ML Components**:
  - Azure OpenAI for LLM capabilities
  - Langchain for RAG pipeline
  - FAISS for vector storage
- **Infrastructure**:
  - Azure App Service for hosting
  - Docker containerization
- **Additional Libraries**:
  - uvicorn for ASGI server
  - python-multipart for file handling
  - pypdf for PDF processing
  - faiss-cpu for vector operations

## 📋 Prerequisites

- Python 3.10+
- Azure subscription with OpenAI access
- Azure CLI (for deployment)

## ⚙️ Environment Variables

Create a `.env` file with:

```env
OPENAI_API_VERSION=your-api-version
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your-chat-deployment
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=your-embeddings-deployment
SECURITY_API_KEY=your-api-key
```

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bajaj-hackathon-api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   uvicorn main:app --reload
   ```

## 🐳 Docker Deployment

1. **Build the container**
   ```bash
   docker build -t insurance-qa-api .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 insurance-qa-api
   ```

## 📡 API Endpoints

### POST /hackrx/run
Process multiple questions against a policy document.

**Request Body**:
```json
{
    "documents": "url-to-policy-document",
    "questions": ["question1", "question2"]
}
```

**Response**:
```json
{
    "answers": ["answer1", "answer2"]
}
```

## 🚀 Azure Deployment

1. **Login to Azure**
   ```bash
   az login
   ```

2. **Deploy to Azure App Service**
   ```bash
   az webapp up --name bajaj-api --resource-group bajaj-hackathon --runtime "PYTHON:3.10"
   ```

## 🔒 Security

- API key authentication required for all endpoints
- CORS middleware configured
- Gzip compression for better performance
- Secure handling of environment variables

## 🏗️ Project Structure

```
├── app/
│   ├── __init__.py
│   ├── models.py          # Pydantic models
│   ├── rag_pipeline.py    # RAG implementation
│   └── security.py        # API security
├── main.py               # FastAPI application
├── startup.py           # Azure startup script
├── Dockerfile
├── requirements.txt
└── azure.yaml
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- Azure OpenAI team for the powerful AI capabilities
- FastAPI for the excellent web framework
- Langchain for the RAG implementation framework
