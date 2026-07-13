import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import tools_1
import description

load_dotenv()

client = OpenAI(
    api_key=os.getenv("api_key"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-20b"


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


def run_agent(question, max_steps=12):
    messages = [
        {"role": "system", "content": description.SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    total_tokens = 0
    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=description.tools,
        )
        total_tokens += response.usage.total_tokens
        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)
            for call in message.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)
                print(f"  [agent] {name}({args})")
                if name == "search_arxiv":
                    result = tools_1.search_arxiv(**args)
                elif name == "ingest_paper":
                    result = tools_1.ingest_paper(**args)
                elif name == "retrieve":
                    result = tools_1.retrieve(**args)
                else:
                    result = f"Unknown tool: {name}"
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": format_tool_result(result),
                })
        else:
            print(f"\n[tokens used: {total_tokens}]")
            return message.content

    return "Agent hit step limit."


if __name__ == "__main__":
    query = "Can you tell me about U-Nets: what they are, what they're used for, and briefly how they work?"
    print(run_agent(query))








