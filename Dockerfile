# Use official Python 3.10 image (Linux-based)
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies needed by faiss and your project
RUN apt-get update && apt-get install -y \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose the port your app runs on
EXPOSE 8000

# Startup command (edit if needed)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]