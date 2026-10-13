"""
REGULATORY & COMPLIANCE AGENT

Third specialist agent. Job: given a product, find UK regulations that
would apply to selling/importing it, and structure them clearly.

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
        "One or more API keys are missing from your .env file. "
        "Check TAVILY_API_KEY and OPENAI_API_KEY are both set. "
        "(This agent does not use Companies House.)"
    )

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


OFFICIAL_DOMAIN_HINTS = ["gov.uk", "hse.gov.uk", "ico.org.uk", "trade.gov.uk"]



def discover_regulations(target_product: str, max_results_per_query: int = 6) -> list[dict]:
    
    queries = [
        f"UK regulations importing {target_product}",
        f"UKCA marking requirements enterprise IT hardware",
        f"WEEE regulations UK enterprise servers electronic waste",
        f"UK data protection requirements enterprise hardware data centre",
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




def structure_regulations(raw_results: list[dict], target_product: str) -> list[dict]:
    combined_text = "\n\n".join(
        f"SOURCE: {r['url']}\nTITLE: {r['title']}\nCONTENT: {r['content'][:800]}"
        for r in raw_results
    )

    system_prompt = (
        "You are a compliance research assistant extracting UK regulatory "
        "requirements from raw web search text. Respond with only valid JSON, "
        "no other text, no markdown code fences. Be conservative , if the "
        "text is vague or unclear about a requirement, say so rather than "
        "guessing specifics."
    )

    user_prompt = f"""
The target product is: "{target_product}"

From the search results below, extract UK regulatory requirements that
appear relevant to selling, importing, or operating this kind of product
in the UK. For each distinct requirement/regulation found, return an
object with these exact fields:

- "regulation_name": name of the regulation/requirement as stated in the text
- "category": one of "Product Safety", "Environmental", "Data Protection",
  "Import/Export", or "Other"
- "applies_to_summary": one or two sentences on what this requires, based
  only on the text given, do not add detail that isn't in the source text
- "compliance_action_required": a short phrase on what a company would need
  to do, based only on the text
- "source_url": the URL this was found in

Return a JSON array. If the same regulation appears in multiple sources,
merge into one entry. If a source is vague, still include it but keep the
summary vague too rather than inventing specifics.

SEARCH RESULTS:
{combined_text}
"""

    print("  [Stage 1.5] Structuring regulations with GPT-4o...")
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
        regulations = json.loads(raw_text)
    except json.JSONDecodeError:
        print("  [Stage 1.5] WARNING: GPT-4o did not return valid JSON. Raw output was:")
        print(raw_text)
        regulations = []

    print(f"  [Stage 1.5] Structured {len(regulations)} regulation entries")
    return regulations



def flag_regulations(regulations: list[dict]) -> list[dict]:
    
    for r in regulations:
        source_url = r.get("source_url", "").lower()
        is_official_looking = any(domain in source_url for domain in OFFICIAL_DOMAIN_HINTS)
        r["source_looks_official"] = is_official_looking
        r["needs_manual_verification"] = True  # always -- see module docstring

    official_count = sum(1 for r in regulations if r["source_looks_official"])
    print(
        f"  [Stage 2] {len(regulations)} regulation entries flagged for review "
        f"({official_count} from official-looking domains)"
    )
    return regulations



def run_regulatory_compliance(target_product: str) -> list[dict]:
    print(f"\nRunning regulatory & compliance research for: {target_product}\n")

    raw_results = discover_regulations(target_product)
    for r in raw_results:
        print(f"  RAW: {r['title']} -- {r['url']}")
    regulations = structure_regulations(raw_results, target_product)
    final_output = flag_regulations(regulations)

    output_path = OUTPUT_DIR / "regulatory_requirements.json"
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nDone. Saved {len(final_output)} regulation entries to {output_path}")
    print("REMINDER: every entry needs manual verification against gov.uk before being treated as fact.\n")
    return final_output


if __name__ == "__main__":
    target_product = "Dell PowerEdge enterprise servers"
    run_regulatory_compliance(target_product)