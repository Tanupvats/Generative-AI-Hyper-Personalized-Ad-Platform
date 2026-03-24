
FROM python:3.10-slim

# Set non-interactive environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /voiceconversion

# Install system dependencies (FFmpeg is required for audio processing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# We upgrade pip first to handle modern package metadata
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Download the base models (HuBERT, RMVPE, UVR5) during the build process
# This makes the image larger, but allows the container to start instantly
RUN python download_assets.py

# Create directories for dynamic data
RUN mkdir -p temp models indices datasets

# Define volumes so user data persists across container restarts
VOLUME ["/app/datasets", "/app/models", "/app/indices", "/app/temp"]

# Expose the FastAPI port
EXPOSE 8000

# Start the uvicorn server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]