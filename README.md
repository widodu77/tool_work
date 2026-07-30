# arXiv Research Agent

A multi-stage RAG system over a corpus of arXiv papers, with a live pipeline visualization.
Ask a question and watch it retrieve, rerank, and answer — and if the corpus doesn't cover it,
watch it go fetch a paper and try again.

**🔗 Live demo:** https://huggingface.co/spaces/widodu/arxiv-research-tool

## What it does
- **Hybrid retrieval** over ~90 papers: dense embeddings + BM25 keyword search, fused with
  Reciprocal Rank Fusion, then reranked by a cross-encoder.
- **Grounded answers** streamed token-by-token, with the source papers cited.
- **Live fallback:** when the corpus can't answer, it reformulates the query, searches OpenAlex,
  ingests a fresh paper, and retries — all visualized stage by stage.
- Built from scratch (no LangChain), served with FastAPI + a custom streaming UI.

## Stack
Python · FastAPI (SSE streaming) · Chroma (vector store) · sentence-transformers
(embeddings + cross-encoder reranker) · rank-bm25 · Groq (`openai/gpt-oss-20b`) · OpenAlex API.

## Run locally
```bash
uv sync
# set your Groq key in a .env file:  api_key=your_key_here
uv run uvicorn server:app --reload --port 8000
```
Then open http://localhost:8000.

*(This is a quick README — a fuller writeup with the retrieval evals and design notes is coming.)*
