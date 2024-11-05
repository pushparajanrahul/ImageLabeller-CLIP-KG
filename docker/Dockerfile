# FROM python:3.11-slim
#
# WORKDIR /app
#
# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     git \
#     wget \
#     && rm -rf /var/lib/apt/lists/*
#
# # Copy requirements and install Python packages
# COPY docker/requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
#
# # Create necessary directories
# RUN mkdir -p data/downloaded_images data/labeled_images logs config src
#
# # Copy the source code
# COPY config ./config/
# COPY src ./src/
# COPY main.py .
# COPY .env .
#
# # Set environment variables
# ENV PYTHONPATH=/app
# # ENV MLFLOW_TRACKING_URI=http://mlflow:5000
#
# # Add healthcheck for the application
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD wget --spider -q http://localhost:5000/health || exit 1
#
# # Run the application
# CMD ["python", "main.py"]

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data/downloaded_images data/labeled_images logs config src

# Copy the source code
COPY config ./config/
COPY src ./src/
COPY main.py .
COPY .env .

# Set environment variables
ENV PYTHONPATH=/app

# Add healthcheck for the application
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --spider -q http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "main.py"]