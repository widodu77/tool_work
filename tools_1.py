import urllib.parse
import urllib.request
import feedparser
from pypdf import PdfReader
import chromadb

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
    local_path = f"{paper_id}.pdf".replace("/", "_")
    with urllib.request.urlopen(pdf_url) as response, open(local_path, "wb") as f:
        f.write(response.read())

    # Extract + chunk
    reader = PdfReader(local_path)
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
    results = corpus.query(query_texts=[query], n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    if not docs:
        return "No relevant chunks found in the corpus. Ingest a paper first."
    return "\n\n".join(f"[from: {m['title']}]\n{d}" for d, m in zip(docs, metas))