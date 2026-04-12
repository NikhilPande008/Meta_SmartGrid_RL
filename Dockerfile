FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports for both local and HF
EXPOSE 8000
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Start the API server
CMD ["python", "inference.py"]