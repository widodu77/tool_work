import urllib.parse
import urllib.request
import feedparser
from pypdf import PdfReader
import chromadb
from sentence_transformers import CrossEncoder
import io 
from rank_bm25 import BM25Okapi

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")   

chroma_client = chromadb.PersistentClient(path="./chroma_db")
corpus = chroma_client.get_or_create_collection(name="arxiv_corpus")


# --- cached corpus snapshot + BM25 index ---
# Rebuilt only when the corpus size changes (i.e. after an ingest), so we don't
# re-pull 10K chunks and rebuild BM25 on every single query.
_index_cache = {"count": None}

def get_corpus_index():
    count = corpus.count()
    if _index_cache["count"] != count:
        data = corpus.get()
        ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
        _index_cache.update({
            "count": count,
            "ids": ids,
            "id_to_doc": dict(zip(ids, docs)),
            "id_to_meta": dict(zip(ids, metas)),
            "bm25": BM25Okapi([d.lower().split() for d in docs]) if docs else None,
        })
    return _index_cache


def search_arxiv(title, max_results=3):
    encoded_title = urllib.parse.quote(title)
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_title}&start=0&max_results={max_results}"
    raw = urllib.request.urlopen(url).read()
    feed = feedparser.parse(raw)

    papers = []
    for entry in feed.entries[:max_results]:
        pdf_url = next((link.href for link in entry.links if link.get("title") == "pdf"), None)
        paper_id = entry.id.rsplit("/", 1)[-1] if getattr(entry, "id", None) else None
        papers.append({
            "title": entry.title,
            "abstract": entry.summary,
            "pdf_url": pdf_url,
            "id": paper_id,
        })
    return papers


def chunking(text, size=700, overlap=100):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    return [text[start:start + size] for start in range(0, len(text), step)]


def ingest_paper(pdf_url, paper_id, title):
    # Skip if this paper is already in the corpus
    existing = corpus.get(where={"paper_id": paper_id}, limit=1)
    if existing["ids"]:
        return f"'{title}' is already in the corpus."

    # Download the PDF
    with urllib.request.urlopen(pdf_url) as response:
        pdf_bytes = response.read()


    # Extract + chunk
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = chunking(full_text)
    if not chunks:
        return f"Could not extract any text from '{title}'."

    # Add to the shared corpus with source metadata + globally-unique ids
    corpus.add(
        documents=chunks,
        ids=[f"{paper_id}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"paper_id": paper_id, "title": title, "chunk_index": i}
                   for i in range(len(chunks))],
    )
    return f"Ingested '{title}' ({len(chunks)} chunks)."

def retrieve_chunks(query, k=5, dense_k=20, sparse_k=20, fused_k=20,
                    use_sparse=True, use_rerank=True):
    results = corpus.query(query_texts=[query], n_results=dense_k)
    dense_ids = results["ids"][0]
    if not dense_ids:
        return []

    idx = get_corpus_index()
    id_to_doc  = idx["id_to_doc"]
    id_to_meta = idx["id_to_meta"]

    # candidate pool: hybrid (dense+sparse) or dense only
    if use_sparse:
        sparse_ids = build_sparse_ids(query, top_n=sparse_k)
        cand_ids = rrf(dense_ids, sparse_ids)[:fused_k]
    else:
        cand_ids = dense_ids[:fused_k]

    cand_docs  = [id_to_doc[i]  for i in cand_ids if i in id_to_doc]
    cand_metas = [id_to_meta[i] for i in cand_ids if i in id_to_meta]
    if not cand_docs:
        return []

    # optional rerank
    if use_rerank:
        scores = reranker.predict([(query, d) for d in cand_docs])
        ranked = sorted(zip(scores, cand_docs, cand_metas), key=lambda x: x[0], reverse=True)[:k]
        return [{"score": float(s), "doc": d, "title": m["title"], "paper_id": m["paper_id"]}
                for s, d, m in ranked]
    else:
        top = list(zip(cand_docs, cand_metas))[:k]
        return [{"score": None, "doc": d, "title": m["title"], "paper_id": m["paper_id"]}
                for d, m in top]
    
def retrieve(query, k=3):                      # top-3 won on the eval (hybrid+rerank, Ans@3 = 4.18)
    chunks = retrieve_chunks(query, k)
    if not chunks:
        return "No relevant chunks found in the corpus. Ingest a paper first."
    return "\n\n".join(f"[from: {c['title']}]\n{c['doc']}" for c in chunks)


def build_sparse_ids(query, top_n=20):
    idx = get_corpus_index()
    if idx["bm25"] is None:
        return []
    ids = idx["ids"]
    bm25_scores = idx["bm25"].get_scores(query.lower().split())
    return [ids[i] for i in sorted(range(len(ids)),
                                   key=lambda i: bm25_scores[i], reverse=True)[:top_n]]

def rrf(dense_ids, sparse_ids, k_const=60):
    scores = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k_const + rank)
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k_const + rank)
    return sorted(scores, key=scores.get, reverse=True)

def ab_test(query, k=5):
    def preview(doc):
        return " ".join(doc.split())[:90]

    print(f"\n=== QUERY: {query} ===")

    # A — dense only (embeddings, no BM25, no rerank)
    dense = corpus.query(query_texts=[query], n_results=k)
    print("\n--- DENSE ONLY (embeddings) ---")
    for i, (doc, meta) in enumerate(zip(dense["documents"][0], dense["metadatas"][0]), 1):
        print(f"{i}. [{meta['title'][:35]}] {preview(doc)}")

    # B — full hybrid: dense + BM25 -> RRF -> rerank
    dense_ids = corpus.query(query_texts=[query], n_results=20)["ids"][0]
    idx = get_corpus_index()
    sparse_ids = build_sparse_ids(query, top_n=20)
    id_to_doc = idx["id_to_doc"]
    id_to_meta = idx["id_to_meta"]
    fused_ids = rrf(dense_ids, sparse_ids)[:20]
    fused_docs = [id_to_doc[i] for i in fused_ids if i in id_to_doc]
    fused_metas = [id_to_meta[i] for i in fused_ids if i in id_to_meta]
    scores = reranker.predict([(query, d) for d in fused_docs])
    ranked = sorted(zip(scores, fused_docs, fused_metas), key=lambda x: x[0], reverse=True)[:k]
    print("\n--- HYBRID + RERANK ---")
    for i, (score, doc, meta) in enumerate(ranked, 1):
        print(f"{i}. [score {score:5.2f}] [{meta['title'][:35]}] {preview(doc)}")


import time

def bulk_ingest(arxiv_ids):
    """Fetch metadata for a list of arXiv IDs and ingest each into the corpus."""
    id_list = ",".join(arxiv_ids)
    url = f"http://export.arxiv.org/api/query?id_list={id_list}&max_results={len(arxiv_ids)}"
    raw = urllib.request.urlopen(url).read()
    feed = feedparser.parse(raw)

    for entry in feed.entries:
        paper_id = entry.id.rsplit("/", 1)[-1] if getattr(entry, "id", None) else None
        title = entry.title
        pdf_url = next((link.href for link in entry.links if link.get("title") == "pdf"), None)
        if not pdf_url:
            print(f"  skip {paper_id}: no pdf link")
            continue
        print(ingest_paper(pdf_url, paper_id, title))   # prints "Ingested ..." or "already in corpus"
        time.sleep(3)   # be polite to arXiv between PDF downloads


def bulk_ingest_by_search(queries, per_query=4):
    """Search arXiv for each topic and ingest the top results.
    A fast, ID-free way to grow a diverse corpus."""
    for q in queries:
        print(f"--- searching: {q} ---")
        for p in search_arxiv(q, max_results=per_query):
            if not p["pdf_url"]:
                continue
            print(ingest_paper(p["pdf_url"], p["id"], p["title"]))
            time.sleep(3)   # be polite to arXiv between PDF downloads