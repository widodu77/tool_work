import os
import json
from openai import OpenAI
import tools_1
from dotenv import load_dotenv
import description

load_dotenv()

client = OpenAI(
    api_key=os.getenv("api_key"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-20b"  # strong tool-caller; verify name at console.groq.com/docs/models

"what are the Surface effects on the electronic energy loss of charged particles entering a metal surface?"
"How do transformers use attention, and how is it applied in vision?"
query ="can you tell me about u-nets, what are they, used for, and briefly describe how they work"

messages = [
    {"role": "system", "content": description.SYSTEM_PROMPT},
    {"role": "user", "content": query},
]

def format_tool_result(result):
    if isinstance(result, list) and result and isinstance(result[0], dict):
        lines = []
        for index, item in enumerate(result, start=1):
            title = item.get("title", "<no title>")
            abstract = item.get("abstract", "")
            pdf_url = item.get("pdf_url", "<no pdf_url>")
            paper_id = item.get("id")
            if len(abstract) > 300:
                abstract = abstract[:300].rstrip() + "..."
            lines.append(f"{index}. {title}")
            if paper_id:
                lines.append(f"   id: {paper_id}")
            lines.append(f"   abstract: {abstract}")
            lines.append(f"   pdf_url: {pdf_url}")
            lines.append("")
        return "\n".join(lines).strip()
    return str(result)


def research_paper(question, max_steps=12):
    """Your existing agent — now a function that returns the final answer."""
    messages = [
        {"role": "system", "content": description.SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=description.tools,
        )
        message = response.choices[0].message
        if message.tool_calls:
            messages.append(message)
            for call in message.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)
                print(f"  [researcher] {name}({args})")
                if name == "ingest_paper":
                    result = tools_1.ingest_paper(**args)
                elif name == "search_arxiv":
                    result = tools_1.search_arxiv(**args)
                elif name == "retrieve":
                    result = tools_1.retrieve(**args)
                else:
                    result = f"Unknown tool: {name}"
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": format_tool_result(result),
                })
        else:
            return message.content         
    return "Researcher hit step limit."


def orchestrate(user_question, max_steps=6):
    messages = [
        {"role": "system", "content": description.ORCHESTRATOR_PROMPT},
        {"role": "user", "content": user_question},
    ]
    total_tokens = 0
    for step in range(max_steps):
        
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=description.orchestrator_tools,
        )
        total_tokens+= response.usage.total_tokens
        print(f" tokens for this call = {response.usage.total_tokens}, and total {total_tokens}")
        message = response.choices[0].message
        if message.tool_calls:
            messages.append(message)
            for call in message.tool_calls:
                args = json.loads(call.function.arguments)
                print(f"[orchestrator] delegating: {args['question']}")
                result = research_paper(args["question"])   # <-- a WHOLE agent runs here
                messages.append({
                    "role": "tool", "tool_call_id": call.id, "content": result,
                })
        else:
            return message.content
    return "Orchestrator hit step limit."

print(orchestrate(query))








