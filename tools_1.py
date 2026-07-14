import urllib.parse
import urllib.request
import feedparser
from pypdf import PdfReader
import chromadb
from sentence_transformers import CrossEncoder
import io 

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")   # small, standard, free

chroma_client = chromadb.PersistentClient(path="./chroma_db")   # persists across runs
corpus = chroma_client.get_or_create_collection(name="arxiv_corpus")


def search_arxiv(title):
    encoded_title = urllib.parse.quote(title)
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_title}&start=0&max_results=3"
    raw = urllib.request.urlopen(url).read()
    feed = feedparser.parse(raw)

    papers = []
    for entry in feed.entries[:3]:
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


def retrieve(query, k=5):
    results = corpus.query(query_texts=[query], n_results=20)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return "No relevant chunks found in the corpus. Ingest a paper first."

    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)  # e.g. array([8.2, -3.1, 5.7, ...]) — higher = more relevant

    ranked = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)
    top = ranked[:k]

    return "\n\n".join(f"[from: {meta['title']}]\n{doc}" for _, doc, meta in top)

def ab_test(query, k=5):
    results = corpus.query(query_texts=[query], n_results=20)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    def preview(doc):
        return " ".join(doc.split())[:90]      # collapse whitespace, first 90 chars

    print(f"\n=== QUERY: {query} ===")

    # A — raw embedding order (bi-encoder only; Chroma returns closest-first)
    print("\n--- RAW (embeddings only) ---")
    for i, (doc, meta) in enumerate(zip(docs[:k], metas[:k]), 1):
        print(f"{i}. [{meta['title'][:40]}] {preview(doc)}")

    # B — reranked (cross-encoder over all 20, keep top k)
    scores = reranker.predict([(query, d) for d in docs])
    ranked = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)[:k]
    print("\n--- RERANKED (cross-encoder) ---")
    for i, (score, doc, meta) in enumerate(ranked, 1):
        print(f"{i}. [score {score:5.2f}] [{meta['title'][:40]}] {preview(doc)}")


