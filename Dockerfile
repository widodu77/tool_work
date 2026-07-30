FROM python:3.11-slim

# git is required by HF Spaces "Dev Mode" tooling — install as root before dropping to the user
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# HF Spaces run the container as uid 1000 — create that user so files/caches are writable
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# app code, index.html, and the pre-built chroma_db corpus
COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
