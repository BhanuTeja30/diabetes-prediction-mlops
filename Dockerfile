# Streamlit frontend. Talks to the API over HTTP (API_URL); does not load the model itself.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app
COPY requirements-frontend.txt .
RUN pip install -r requirements-frontend.txt

# The frontend imports the shared Pydantic schemas and config loader.
COPY src/__init__.py src/common.py src/
COPY app/ app/
COPY configs/ configs/

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app/frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
