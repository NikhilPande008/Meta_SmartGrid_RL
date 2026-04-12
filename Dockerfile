FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Expose the standard HF port
EXPOSE 7860

# Launch using the module string for better stability
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]