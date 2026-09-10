FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install system headers, tools, Cairo dev libraries, and GObject introspection dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libdbus-1-dev \
    libglib2.0-dev \
    libgirepository1.0-dev \
    libcairo2-dev \
    gobject-introspection \
    python3-apt \
    dbus \
    meson \
    ninja-build \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements first
COPY requirements.txt .

# 3. Install Python dependencies with standard build isolation
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy application files
COPY . .

EXPOSE 8080

CMD ["python", "main.py"]