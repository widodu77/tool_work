import urllib.request
import feedparser
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np 
import ollama

url = "http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=2"
raw = urllib.request.urlopen(url).read()
feed = feedparser.parse(raw)
entry = feed.entries[1]


pdf_url = next(link.href for link in entry.links if link.get("title") == "pdf")
with urllib.request.urlopen(pdf_url) as response, open("paper.pdf", "wb") as f:
    f.write(response.read())

#---------

reader = PdfReader("paper.pdf")
pages = [page.extract_text() or "" for page in reader.pages]
full_text = "\n".join(pages)

def chunking(text, size=800, overlap=100):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    return [text[start:start + size] for start in range(0, len(text), step)]

chunks = chunking(full_text)
#print(f"created {len(chunks)} chunks")
#print(chunks[0])

model = SentenceTransformer("all-MiniLM-L6-v2")

encoded = model.encode(chunks)

def cosine_sim(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return np.dot(a, b) / denom

def similarities(matrix, query):
    query = np.asarray(query)
    if query.ndim == 2 and query.shape[0] == 1:
        query = query[0]

    return np.array([cosine_sim(row, query) for row in matrix])


#query + embedding + similiratiy chunk retrieval

query = " what are the Surface effects on the electronic energy loss of charged particles entering a metal surface?"
query2= "what is the capital of france?"

query_encoded = model.encode(query)

sims = similarities(encoded, query_encoded)
top = sims.argsort()[::-1][:5]   

#for i in top:
#    print(f"\n[score {sims[i]:.3f}] chunk {i}")
#    print(chunks[i])

top_text = "\n".join(chunks[i] for i in top)

#prompt building 

prompt = (
    "Use ONLY the context below to answer. If the answer isn't in it, say I don't know.\n"
    f"Context: {top_text}\n"
    f"Question: {query2}"
)

messages = [{"role": "user", "content": prompt}]
response = ollama.chat(model="gemma4", messages=messages)

print(response.message.content)


