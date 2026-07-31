FROM python:3.11-slim

WORKDIR /app

# System dependencies for TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ ./src/
COPY api/ ./api/
COPY results/models/ ./results/models/
COPY data/ ./data/

# Expose inference port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
