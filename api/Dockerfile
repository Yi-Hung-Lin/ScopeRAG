FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8021
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8021"]
