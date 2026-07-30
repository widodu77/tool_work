import os
import json
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

from tools_1 import (corpus, get_corpus_index, build_sparse_ids, rrf, reranker,
                     search_openalex, ingest_fetched_paper)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("api_key"),
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "openai/gpt-oss-20b"


def sse(obj):
    """Format a dict as one Server-Sent Event."""
    return f"data: {json.dumps(obj)}\n\n"


MISS_THRESHOLD = 0.0   # top cross-encoder score below this ⇒ corpus doesn't have the answer (tunable)


def retrieval_stages(question):
    """Run dense → sparse → fuse → rerank, yielding ('sse', event) per stage,
    then ('result', ranked_top3) at the end."""
    dense = corpus.query(query_texts=[question], n_results=20)
    dense_ids = dense["ids"][0]
    if not dense_ids:
        yield ("result", [])
        return
    yield ("sse", sse({"type": "stage", "stage": "dense", "detail": {"candidates": len(dense_ids)}}))

    idx = get_corpus_index()
    id_to_doc, id_to_meta = idx["id_to_doc"], idx["id_to_meta"]

    sparse_ids = build_sparse_ids(question, top_n=20)
    yield ("sse", sse({"type": "stage", "stage": "sparse", "detail": {"candidates": len(sparse_ids)}}))

    fused_ids = rrf(dense_ids, sparse_ids)[:20]
    yield ("sse", sse({"type": "stage", "stage": "fused", "detail": {"pool": len(fused_ids)}}))

    cand_docs  = [id_to_doc[i]  for i in fused_ids if i in id_to_doc]
    cand_metas = [id_to_meta[i] for i in fused_ids if i in id_to_meta]

    scores = reranker.predict([(question, d) for d in cand_docs])
    ranked = sorted(zip(scores, cand_docs, cand_metas), key=lambda x: x[0], reverse=True)[:3]
    top_chunks = [
        {"title": m["title"], "score": round(float(s), 2), "preview": " ".join(d.split())[:200]}
        for s, d, m in ranked
    ]
    yield ("sse", sse({"type": "stage", "stage": "reranked", "detail": {"chunks": top_chunks}}))
    yield ("result", ranked)


def make_search_query(question):
    """Turn a natural-language question into a focused arXiv keyword query."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content":
                "Convert the user's question into a short arXiv search query: 3-6 key "
                "technical terms, no punctuation, no quotes. Respond with ONLY the query."},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content.strip().strip('"')


def pipeline_events(question, force=False):
    """Retrieve from the corpus; if it comes up empty-handed (or `force`), go search
    arXiv, ingest a paper, and retry — emitting an SSE event at every step."""
    try:
        ranked = []
        for kind, val in retrieval_stages(question):
            if kind == "sse":
                yield val
            else:
                ranked = val

        if not ranked:
            yield sse({"type": "error", "message": "Corpus is empty — ingest papers first."})
            return

        # --- corpus miss (or forced): fetch a fresh paper and retry ---
        if force or ranked[0][0] < MISS_THRESHOLD:
            yield sse({"type": "corpus_miss", "top_score": round(float(ranked[0][0]), 2),
                       "forced": force})
            try:
                # 1. reformulate the question into a focused keyword query
                search_q = make_search_query(question)
                yield sse({"type": "reformulate", "original": question, "query": search_q})

                # 2. search OpenAlex (keyless, tolerant of cloud IPs, unlike arXiv / S2)
                papers = search_openalex(search_q, limit=5)
                yield sse({"type": "stage", "stage": "searching",
                           "detail": {"papers": [p["title"] for p in papers]}})

                # 3. ingest the paper whose abstract best matches (full PDF if available, else abstract)
                ingested = None
                if papers:
                    ab_scores = reranker.predict([(question, p["abstract"]) for p in papers])
                    best = papers[max(range(len(papers)), key=lambda i: ab_scores[i])]
                    ingest_fetched_paper(best)
                    ingested = best

                yield sse({"type": "stage", "stage": "ingesting",
                           "detail": {"title": ingested["title"] if ingested else "no paper found"}})

                if ingested:
                    # retry retrieval over the now-bigger corpus
                    for kind, val in retrieval_stages(question):
                        if kind == "sse":
                            yield val
                        else:
                            ranked = val
            except Exception:
                traceback.print_exc()   # real cause shows in the HF Space "Logs" tab
                # live source rate-limited / unavailable — degrade gracefully instead of crashing
                yield sse({"type": "fetch_failed", "message":
                    "The live paper lookup is unavailable right now. Answering from the existing corpus instead."})

        # --- answer, grounded in the (possibly refreshed) top chunks, streamed ---
        context = "\n\n".join(d for _, d, _ in ranked)
        answer_stream = client.chat.completions.create(
            model=MODEL,
            stream=True,
            messages=[
                {"role": "system", "content":
                    "Answer the question using ONLY the context provided. Cite the paper "
                    "title(s) you draw from. If the context lacks the answer, say you don't know."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        for chunk in answer_stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield sse({"type": "answer_delta", "text": delta})

        sources = sorted(set(m["title"] for _, _, m in ranked))
        yield sse({"type": "done", "sources": sources})

    except Exception as e:
        yield sse({"type": "error", "message": str(e)})


@asynccontextmanager
async def lifespan(app):
    get_corpus_index()   # pre-warm BM25 + corpus snapshot so the first query is fast
    print("Corpus index pre-warmed.")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/ask")
def ask(q: str, force: bool = False):
    """Stream the retrieval pipeline + answer for a question, as SSE events."""
    return StreamingResponse(pipeline_events(q, force), media_type="text/event-stream")


@app.get("/")
def root():
    return FileResponse("index.html")


@app.get("/status")
def status():
    idx = get_corpus_index()
    papers = len(set(m["paper_id"] for m in idx["id_to_meta"].values()))
    return {"status": "ok", "chunks": idx["count"], "papers": papers}
