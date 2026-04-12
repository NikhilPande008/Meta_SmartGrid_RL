FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# MUST EXPOSE 8000
EXPOSE 8000
# MUST RUN inference.py
CMD ["python", "inference.py"]