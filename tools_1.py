import urllib.parse
import urllib.request  
import ollama  
import feedparser 
from pypdf import PdfReader 
from sentence_transformers import SentenceTransformer  
import numpy as np  
import chromadb

# High-level pipeline comments:
# - retrieve(): query arXiv and download a PDF called "paper.pdf"
# - parse(): read a PDF, chunk the text, embed chunks, and return top chunks matching a query

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
paper_embedding_cache = {}

def search_arxiv(title):
    encoded_title = urllib.parse.quote(title)
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_title}&start=0&max_results=3"
    raw = urllib.request.urlopen(url).read()
    feed = feedparser.parse(raw)

    papers = []
    for entry in feed.entries[:5]:
        pdf_url = next((link.href for link in entry.links if link.get("title") == "pdf"), None)
        paper_id = entry.id.rsplit("/", 1)[-1] if getattr(entry, "id", None) else None
        papers.append({
            "title": entry.title,
            "abstract": entry.summary,
            "pdf_url": pdf_url,
            "id": paper_id,
        })

    return papers

chroma_client = chromadb.PersistentClient(path="./chroma_db")  # saves to disk

corpus = chroma_client.get_or_create_collection(name="arxiv_corpus")

def ingest_paper(paper_name, paper_id, title):
    # 1. Skip if this paper is already in the corpus
    existing = corpus.get(where={"paper_id": paper_id}, limit=1)
    if existing["ids"]:
        return f"{title} already in corpus."
    
    def chunking(text, size=700, overlap=100):
        
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        step = size - overlap  
        return [text[start:start + size] for start in range(0, len(text), step)]

    # 2. Extract + chunk the PDF text
    pdf_path = paper_name if paper_name.lower().endswith(".pdf") else f"{paper_name}.pdf"
    reader = PdfReader(pdf_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = chunking(full_text)

    # 3. Add to the shared corpus WITH source metadata + unique ids
    corpus.add(
        documents=chunks,
        ids=[f"{paper_id}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"paper_id": paper_id, "title": title, "chunk_index": i}
                   for i in range(len(chunks))],
    )
    return f"Ingested '{title}' ({len(chunks)} chunks)."

def retrieve(query, k=5):
    results = corpus.query(query_texts=[query], n_results=k)
    docs  = results["documents"][0]
    metas = results["metadatas"][0]      # <-- source info comes back here
    # format so the model sees which paper each chunk is from:
    return "\n\n".join(
        f"[from: {m['title']}]\n{d}" for d, m in zip(docs, metas)
    )





