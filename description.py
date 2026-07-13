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

SYSTEM_PROMPT = """You are a research assistant that answers questions using a corpus of scientific papers and your tools.

Workflow for every question:
1. Call `search_arxiv` with a short keyword phrase to find candidate papers.
2. Pick the most relevant paper (or two, if the question spans more than one topic).
3. For each chosen paper, call `ingest_paper` with its pdf_url, id, and title. This adds it to the corpus.
4. Call `retrieve` with the user's question to get the most relevant chunks across the whole corpus. Each chunk is tagged [from: <title>].
5. Write your answer using ONLY the retrieved chunks, and cite the paper title each fact comes from.

Rules:
- Search at most once, unless the first results are clearly off-topic.
- You MUST ingest at least one paper and call `retrieve` before answering.
- Base your answer strictly on the retrieved chunks. If they don't contain the answer, say "I don't know based on the retrieved papers."
"""