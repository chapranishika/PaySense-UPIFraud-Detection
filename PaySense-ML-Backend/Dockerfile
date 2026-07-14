FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY paysense_pipeline.py .
COPY paysense_ml_pipeline.py .
COPY paysense_phase3.py .
COPY main.py .

# If artefacts/ directory exists (pre-built model), copy it
# Otherwise the startup script will generate it
COPY artefacts/ artefacts/ 2>/dev/null || true

EXPOSE 8000

# Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
