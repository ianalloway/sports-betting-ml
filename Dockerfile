FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# No training at build time (issue #28): training in the image build made
# builds slow and non-reproducible. The entrypoint uses a CI-trained
# artifact mounted at /app/model/artifacts and only trains a fallback
# model at startup when no artifact is present.

# Expose port
EXPOSE 7860

ENTRYPOINT ["./docker/entrypoint.sh"]
