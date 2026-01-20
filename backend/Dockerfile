FROM python:3.12-slim

# Install system dependencies for Playwright and VNC
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    # VNC dependencies
    xvfb \
    x11vnc \
    fluxbox \
    novnc \
    websockify \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for faster package installation
RUN pip install uv

# Copy backend requirements first (for caching)
COPY backend/pyproject.toml backend/
COPY backend/README.md backend/

# Install backend dependencies
RUN cd backend && uv pip install -e . --system

# Copy and install openbrowser from local path
# Set version via environment variable since .git is not available
COPY pyproject.toml .
COPY src/ src/
COPY README.md .
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.2.94
RUN uv pip install -e . --system

# Install Playwright and browsers (chromium works better in Docker)
RUN playwright install chromium
RUN playwright install-deps chromium

# Set environment variable to tell openbrowser where the browser is
ENV PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright
ENV BROWSER_PATH=/root/.cache/ms-playwright/chromium-1200/chrome-linux/chrome

# Copy backend code
COPY backend/ backend/

WORKDIR /app/backend

# Create directories and set permissions BEFORE switching user
RUN mkdir -p /home/appuser/.cache/ms-playwright && \
    mkdir -p /home/appuser/.config && \
    mkdir -p /home/appuser/.local && \
    useradd -m -u 1000 appuser && \
    cp -r /root/.cache/ms-playwright/* /home/appuser/.cache/ms-playwright/ && \
    chown -R appuser:appuser /app /home/appuser

USER appuser

# Update environment for appuser
ENV PLAYWRIGHT_BROWSERS_PATH=/home/appuser/.cache/ms-playwright
ENV HOME=/home/appuser

# Expose ports: 8000 for API, 6080+ for websockify VNC
EXPOSE 8000
EXPOSE 6080-6090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
