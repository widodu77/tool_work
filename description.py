tools = [
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": "Search arXiv for papers matching a keyword phrase. Returns candidate papers, each with a title, abstract, pdf_url, and id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Keyword phrase or paper title to search for."}
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_paper",
            "description": "Download a paper's PDF and add it to the shared corpus so it can be searched. Call once per paper you want to read.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pdf_url": {"type": "string", "description": "The paper's pdf_url, taken from search_arxiv results."},
                    "paper_id": {"type": "string", "description": "The paper's id, taken from search_arxiv results."},
                    "title": {"type": "string", "description": "The paper's title, taken from search_arxiv results."},
                },
                "required": ["pdf_url", "paper_id", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "Search the shared corpus of ingested papers and return the most relevant chunks. Each chunk is tagged with the paper it came from.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question or topic to search the corpus for."},
                    "k": {"type": "integer", "description": "How many chunks to return (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a research assistant that answers questions from a persistent corpus of scientific papers. The corpus is your PRIMARY source; arXiv search is only for filling gaps.

Workflow for every question:
1. FIRST call `retrieve` with the question to see what the corpus already contains.
# in SYSTEM_PROMPT, replace step 2 and the search rule:
2. Look at what `retrieve` returned:
   - If it returned the message "No relevant chunks found in the corpus", THEN search:
     call `search_arxiv` once, ingest 1-2 papers, and call `retrieve` again.
   - Otherwise it returned real chunks — you MUST answer from them. Do NOT search.
3. Never ingest a new paper if the corpus already has relevant material on the topic.
4. Answer using ONLY the retrieved chunks, citing the paper title each fact came from.

Rules:
- Only call `search_arxiv` when `retrieve` returned the "No relevant chunks found" message.
  If `retrieve` returned ANY real chunks, never search — answer from what you have.earch_arxiv` ONCE, pick the 1-2 most relevant papers, `ingest_paper` each, and call `retrieve` again.
- Always retrieve BEFORE searching. Check the corpus first; arXiv is the fallback.
- Base your answer strictly on the retrieved chunks. If they don't contain the answer, say "I don't know based on the corpus."
"""