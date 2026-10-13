"""
BUSINESS PROCESS AGENT

Fourth specialist agent. Job: given a product, research and structure a
typical UK market-entry PROCESS for it, the steps a company would go
through to actually launch and sell it in the UK.

DIFFERENT IN KIND FROM THE OTHER THREE AGENTS:
The other agents find and rank a LIST of things (competitors, distributors,
regulations). This one is closer to SYNTHESIS, there usually isn't one
single "correct" process listed on any one webpage, so this agent leans
more heavily on GPT-4o's ability to combine and reason across several
partial descriptions into one coherent, ordered process map.

Stage 1 (discover)  - Tavily search for how companies typically approach
                        UK market entry / channel strategy / B2B sales
                        cycles for hardware like this
Stage 1.5 (synthesize) - GPT-4o combines the (probably partial, possibly
                        inconsistent) search results into ONE ordered
                        sequence of process stages
No Stage 2 filtering, no scikit-learn, no Companies House, this agent
doesn't have a "which of these are real/relevant" problem the way the
discovery agents do. Its main risk is a different one: GPT-4o could
over-generalize or fill gaps with generic business-process knowledge rather
than what the search results actually said. The prompt explicitly asks it
to distinguish between the two, and the output flags which parts came from
sources vs. general reasoning.

LIMITATION: because this is closer to synthesis than retrieval, this
agent's output is more dependent on GPT-4o's general training knowledge of
"how B2B hardware sales work" than the other three agents, which lean more
on the actual retrieved text. this is a real, structural difference in how 
trustworthy/source-grounded this agent's
output is compared to the others.
"""

import os
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from tavily import TavilyClient
from openai import OpenAI



load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([TAVILY_API_KEY, OPENAI_API_KEY]):
    raise EnvironmentError(
        "One or more API keys are missing from .env file. "
        "Check TAVILY_API_KEY and OPENAI_API_KEY are both set. "
        "(This agent does not use Companies House.)"
    )

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


#discover 

def discover_process_info(target_product: str, max_results_per_query: int = 6) -> list[dict]:
    queries = [
        f"{target_product} UK market entry strategy",
        f"enterprise hardware channel partner strategy UK",
        f"B2B sales cycle enterprise IT hardware UK",
    ]

    raw_results = []
    for query in queries:
        print(f"  [Stage 1] Searching: {query}")
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results_per_query,
        )
        for result in response.get("results", []):
            raw_results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                }
            )
        time.sleep(0.5)

    print(f"  [Stage 1] Collected {len(raw_results)} raw results")
    return raw_results


#synthesis

def synthesize_process_map(raw_results: list[dict], target_product: str) -> dict:
    """
    Unlike the other agents, this returns one combined structure (a process
    map), not a list of separate independent entries.
    """
    combined_text = "\n\n".join(
        f"SOURCE: {r['url']}\nTITLE: {r['title']}\nCONTENT: {r['content'][:800]}"
        for r in raw_results
    )

    system_prompt = (
        "You are a business analyst synthesizing a market-entry process map "
        "from partial web search text. Respond with ONLY valid JSON, no other "
        "text, no markdown code fences. Be explicit about which parts are "
        "grounded in the search text versus general business knowledge."
    )

    user_prompt = f"""
The target product is: "{target_product}"

Based on the search results below, construct a typical UK market-entry
process for a product like this, as an ORDERED sequence of stages (e.g.
market research, regulatory clearance, channel partner setup, pilot sales,
full launch, adjust based on what's actually relevant here).

For each stage, return an object with these exact fields:
- "stage_name": short name of this stage
- "description": one or two sentences on what happens in this stage
- "typical_duration": a rough duration if the text gives any signal (e.g.
  "4-8 weeks"), or "Not specified in sources" if there's no evidence for it
- "grounded_in_sources": true if this stage's content is actually reflected
  in the search results below, false if you are filling a gap using general
  business knowledge because the sources didn't cover it

Return a JSON object with a single key "stages" containing an array of these
stage objects, ordered from first to last.

SEARCH RESULTS:
{combined_text}
"""

    print("  [Stage 1.5] Synthesizing process map with GPT-4o...")
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    raw_text = response.choices[0].message.content.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:]

    try:
        result = json.loads(raw_text)
        stages = result.get("stages", [])
    except json.JSONDecodeError:
        print("  [Stage 1.5] WARNING: GPT-4o did not return valid JSON. Raw output was:")
        print(raw_text)
        stages = []

    grounded_count = sum(1 for s in stages if s.get("grounded_in_sources"))
    print(
        f"  [Stage 1.5] Synthesized {len(stages)} process stages "
        f"({grounded_count} grounded in sources, {len(stages) - grounded_count} from general reasoning)"
    )
    return {"target_product": target_product, "stages": stages}



# ORCHESTRATION


def run_business_process(target_product: str) -> dict:
    print(f"\nRunning business process research for: {target_product}\n")

    raw_results = discover_process_info(target_product)
    process_map = synthesize_process_map(raw_results, target_product)

    output_path = OUTPUT_DIR / "business_process_map.json"
    with open(output_path, "w") as f:
        json.dump(process_map, f, indent=2)

    print(f"\nDone. Saved process map with {len(process_map['stages'])} stages to {output_path}\n")
    return process_map


if __name__ == "__main__":
    target_product = "Dell PowerEdge enterprise servers"
    run_business_process(target_product)