tools = [
    {
        "type": "function",
        "function": {
            "name": "parse",
            "description": "Reads a PDF, extracts its text, splits it into chunks, embeds the chunks, and returns the most relevant chunks for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_name": {
                        "type": "string",
                        "description": "The PDF file name to read, such as paper.pdf.",
                    },
                    "main_query": {
                        "type": "string",
                        "description": "The user's question or search query.",
                    },
                },
                "required": ["paper_name", "main_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": "Searches arXiv for papers matching a title query and returns their metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The paper title or keyword phrase to search for.",
                    }
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_from_arx",
            "description": "Downloads a PDF from a provided arXiv URL and saves it under a chosen file name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pdf_url": {
                        "type": "string",
                        "description": "The direct URL of the PDF to download.",
                    },
                    "paper_name": {
                        "type": "string",
                        "description": "The base name to use for the saved PDF file.",
                    },
                },
                "required": ["pdf_url", "paper_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_paper",
            "description": "Ingests a paper PDF into the shared Chroma corpus with chunk metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_name": {
                        "type": "string",
                        "description": "The PDF file name to ingest, such as paper.pdf.",
                    },
                    "paper_id": {
                        "type": "string",
                        "description": "A unique paper identifier used for corpus deduplication.",
                    },
                    "title": {
                        "type": "string",
                        "description": "The paper title to store in metadata.",
                    },
                },
                "required": ["paper_name", "paper_id", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "Searches the shared Chroma corpus and returns the most relevant document chunks for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query used to retrieve relevant chunks.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "The number of top chunks to return.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a research assistant that answers questions about scientific papers using your tools.

Workflow:
1. Call `search_arxiv` ONCE with a short keyword phrase. The results already contain relevant papers.
2. Pick the single most relevant paper from those results. Do NOT search again — the first results are good enough.
3. Call `download_from_arx` with that paper's pdf_url and a short paper_name. It returns the saved filename.
4. Call `parse` with that filename and the question.
5. As soon as `parse` returns chunks, WRITE YOUR ANSWER using only those chunks.

Rules:
- Search AT MOST once. Only search a second time if the first search returned nothing on-topic.
- Once you have parsed a paper, you MUST answer. Never go back to searching after parsing.
- Base your answer strictly on the parsed chunks. If they don't contain the answer, say "I don't know based on this paper."
"""

ORCHESTRATOR_PROMPT = """You are a research coordinator. Break the user's question into focused 
                        sub-questions and delegate each to the `ask_researcher` tool, which investigates
                        one paper and returns a grounded answer. Call `ask_researcher` once per sub-question 
                        (you may call it several times). Then synthesize the sub-answers into one final answer.
                        Do not answer from your own knowledge — rely on the researcher's results.
                        Try to only run a moderate amount of agents, maybe 2 on average"""

orchestrator_tools = [{
    "type": "function",
    "function": {
        "name": "ask_researcher",
        "description": "Delegate a single focused research question to a sub-agent that reads one paper and returns a grounded answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "One focused research question."}
            },
            "required": ["question"],
        },
    },
}]