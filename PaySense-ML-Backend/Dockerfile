FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source. A previous version hand-picked individual files here and
# never included src/ -- main.py's `from src.fraud_model import ...` then
# failed at container startup every time (verified: `docker run` crashed
# with ModuleNotFoundError: No module named 'src'). Copying everything not
# excluded by .dockerignore (venv/, datasets, logs, secrets) avoids this
# whole class of "forgot to list a file" bug.
COPY . .

EXPOSE 8000

# Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
