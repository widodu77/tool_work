import os
import time
import openai
from openai import OpenAI
from dotenv import load_dotenv
from tools_1 import retrieve_chunks
import description

load_dotenv()

client = OpenAI(
    api_key=os.getenv("api_key"),
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "openai/gpt-oss-20b"


def chat(messages):
    """Groq call with simple rate-limit backoff (free tier is ~6K tokens/min)."""
    while True:
        try:
            return client.chat.completions.create(model=MODEL, messages=messages)
        except openai.RateLimitError:
            print("  rate limited, waiting 20s...")
            time.sleep(20)


# ---------- retrieval metrics (paper-level) ----------
def eval_retrieval(golden, k=5, use_sparse=True, use_rerank=True):
    hits, rr = 0, []
    for case in golden:
        chunks = retrieve_chunks(case["question"], k=k,
                                 use_sparse=use_sparse, use_rerank=use_rerank)
        titles = [c["title"] for c in chunks]
        rank = next((i for i, t in enumerate(titles, 1)
                     if case["expected_paper"].lower() in t.lower()), None)
        if rank:
            hits += 1
            rr.append(1 / rank)
        else:
            rr.append(0.0)
    return hits / len(golden), sum(rr) / len(golden)


# ---------- answer quality (LLM-as-judge) ----------
def answer_from_chunks(question, chunks):
    """Answer the question grounded only in the retrieved chunks."""
    context = "\n\n".join(c["doc"] for c in chunks)
    resp = chat([
        {"role": "system", "content":
            "Answer the question using ONLY the context provided. "
            "If the context does not contain the answer, say you don't know."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ])
    return resp.choices[0].message.content


def judge_answer(question, answer):
    """Score an answer 1-5 for correctness + completeness. Returns 0 if unparseable."""
    resp = chat([
        {"role": "system", "content":
            "You grade answers to machine-learning research questions on a 1-5 scale for "
            "correctness and completeness. 5 = fully correct and complete, 3 = partially "
            "correct, 1 = wrong or irrelevant. Respond with ONLY a single integer from 1 to 5."},
        {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}\n\nScore (1-5):"},
    ])
    text = resp.choices[0].message.content.strip()
    for tok in text.replace(".", " ").split():
        if tok.isdigit():
            return int(tok)
    return 0  # unparseable


def eval_answers(golden, use_sparse=True, use_rerank=True):
    """Average answer-quality score across the golden set for one retrieval config."""
    total = 0
    for case in golden:
        chunks = retrieve_chunks(case["question"], k=5,
                                 use_sparse=use_sparse, use_rerank=use_rerank)
        answer = answer_from_chunks(case["question"], chunks)
        total += judge_answer(case["question"], answer)
    return total / len(golden)


if __name__ == "__main__":
    configs = [
        ("dense only",      dict(use_sparse=False, use_rerank=False)),
        ("dense + rerank",  dict(use_sparse=False, use_rerank=True)),
        ("hybrid + rerank", dict(use_sparse=True,  use_rerank=True)),
    ]
    print(f"{'config':<18}{'Recall@5':>10}{'MRR':>8}{'AnswerScore':>13}")
    for name, cfg in configs:
        recall, mrr = eval_retrieval(description.GOLDEN_SET, k=5, **cfg)
        ans = eval_answers(description.GOLDEN_SET, **cfg)
        print(f"{name:<18}{recall:>10.2f}{mrr:>8.2f}{ans:>13.2f}")
