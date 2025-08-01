from duckduckgo_search import DDGS
from typing import List, Tuple

def extract_top_results(query: str, max_results: int = 5) -> List[dict]:
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))

def summarize_results(results: List[dict], llm) -> Tuple[str, List[str]]:
    combined_text = "\n\n".join(f"{r['title']}:\n{r['body']}" for r in results)
    prompt = f"""
You are a helpful research assistant. Read the following search results and summarize a reliable answer.
    
{combined_text}

Then, include a list of clean reference links at the bottom.
"""
    summary = llm.generate(prompt)
    references = [r["href"] for r in results]
    return summary, references
