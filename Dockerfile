# ============================================================
# Spark Distributed Cluster - Custom Image
# Base: Spark 3.5.x with Python 3.11
# Includes: numpy, pandas, sklearn, torch, statsmodels, etc.
# ============================================================

FROM apache/spark:3.5.3-python3

USER root

# Install system dependencies for matplotlib (headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ python3-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python ML dependencies
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt \
    && rm /tmp/requirements-docker.txt

# Set working directory
WORKDIR /app

# Set matplotlib to use non-interactive backend
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1

USER spark
