"""
COMPETITOR ANALYSIS AGENT (v3)

This is the first specialist agent in the GTM system. Its whole job is:
  give it a product, it gives you back a clean list of real UK competitors,
  each with enough structured info that the Report Synthesis Agent can later
  build a market positioning map out of it.

"""

import os
import json
import base64
import time
import difflib
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from tavily import TavilyClient
from openai import OpenAI



load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COMPANIES_HOUSE_API_KEY = os.getenv("COMPANIES_HOUSE_API_KEY")

if not all([TAVILY_API_KEY, OPENAI_API_KEY, COMPANIES_HOUSE_API_KEY]):
    raise EnvironmentError(
        "One or more API keys are missing from your .env file. "
        "Check TAVILY_API_KEY, OPENAI_API_KEY, and COMPANIES_HOUSE_API_KEY are all set."
    )

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)




def discover_candidates(target_product: str, max_results_per_query: int = 6) -> list[dict]:
    
    queries = [
        f"{target_product} competitors UK",
        f"best enterprise server vendors UK 2026",
        f"{target_product} alternatives comparison",
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




def structure_candidates(raw_results: list[dict], target_product: str) -> list[dict]:
   
    combined_text = "\n\n".join(
        f"SOURCE: {r['url']}\nTITLE: {r['title']}\nCONTENT: {r['content'][:800]}"
        for r in raw_results
    )

    system_prompt = (
        "You are a market research assistant extracting competitor companies "
        "from raw web search text. Respond with ONLY valid JSON, no other text, "
        "no markdown code fences."
    )

    user_prompt = f"""
The target product is: "{target_product}"

From the search results below, extract a list of DISTINCT companies that appear
to be competitors selling similar products. Ignore the target product's own
company if it appears. Ignore generic listicle sites, review platforms, or news
aggregators themselves (e.g. do not list "TechRadar" as a competitor).

For each company found, return an object with these exact fields:
- "name": company name as written
- "product_summary": one sentence on what they sell, based only on the text given
- "uk_relevance_evidence": short quote or paraphrase showing UK presence, or "" if none found
- "price_tier": one of "Budget", "Mid-range", "Premium", or "Unknown": base this
  ONLY on how the text describes the product. Do NOT invent a specific price.
- "specialization": a short phrase (a few words) on what this company focuses on.
  Use "Unknown" if unclear.
- "source_url": the URL this was found in

Return a JSON array of these objects. If the same company appears in multiple
sources, merge into a single entry and combine the evidence.

SEARCH RESULTS:
{combined_text}
"""

    print("  [Stage 1.5] Structuring candidates with GPT-4o...")
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
        candidates = json.loads(raw_text)
    except json.JSONDecodeError:
        print("  [Stage 1.5] WARNING: GPT-4o did not return valid JSON. Raw output was:")
        print(raw_text)
        candidates = []

    print(f"  [Stage 1.5] Structured {len(candidates)} candidate companies")
    return candidates




def shortlist_candidates(candidates: list[dict], top_n: int = 10) -> list[dict]:
   
    server_keywords = ["server", "rack", "data centre", "data center", "enterprise hardware"]

    passed_filters = []
    for c in candidates:
        text_blob = f"{c.get('name','')} {c.get('product_summary','')}".lower()
        has_product_overlap = any(kw in text_blob for kw in server_keywords)

        if has_product_overlap:
            passed_filters.append(c)

    print(f"  [Stage 2] {len(passed_filters)} of {len(candidates)} candidates passed the product-overlap filter")

    for c in passed_filters:
        evidence_len = len(c.get("uk_relevance_evidence", ""))
        summary_len = len(c.get("product_summary", ""))
        c["_score"] = evidence_len * 0.7 + summary_len * 0.3

    passed_filters.sort(key=lambda c: c["_score"], reverse=True)
    shortlist = passed_filters[:top_n]

    print(f"  [Stage 2] Shortlisted top {len(shortlist)} candidates")
    return shortlist




def _companies_house_auth_header() -> dict:
    """Companies House uses HTTP Basic Auth: your API key as username, blank password."""
    token = base64.b64encode(f"{COMPANIES_HOUSE_API_KEY}:".encode()).decode()
    return {"Authorization": f"Basic {token}"}


# a small set of UK SIC codes (industry classification codes) related to
# computers/servers/tech hardware, used to check whether a match is
# actually in the right INDUSTRY, not just an active company sharing a name
RELEVANT_SIC_CODES = {
    "26200",  # Manufacture of computers and peripheral equipment
    "46510",  # Wholesale of computers
    "62020",  # Computer consultancy activities
    "62090",  # Other information technology and computer service 
    "46900",  # Non-specialised wholesale trade 
    "47410",  # Retail sale of computers 
}


def _name_similarity(target_name: str, candidate_name: str) -> float:
    
    a = target_name.lower().strip()
    b = candidate_name.lower().strip()
    return difflib.SequenceMatcher(None, a, b).ratio()


def _fetch_company_profile(company_number: str) -> dict:
   
    headers = _companies_house_auth_header()
    try:
        resp = requests.get(
            f"https://api.company-information.service.gov.uk/company/{company_number}",
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return {}


def _score_candidate(target_name: str, candidate: dict) -> float:
   
    name_score = _name_similarity(target_name, candidate.get("title", ""))

    profile = _fetch_company_profile(candidate.get("company_number", ""))
    sic_codes = set(profile.get("sic_codes", []))
    sic_score = 1.0 if sic_codes & RELEVANT_SIC_CODES else 0.0

    date_str = candidate.get("date_of_creation", "")
    if date_str:
        try:
            incorporation_year = int(date_str[:4])
            age_years = datetime.now().year - incorporation_year
            age_score = min(age_years / 15, 1.0)
        except ValueError:
            age_score = 0.0
    else:
        age_score = 0.0

    return (name_score * 0.5) + (sic_score * 0.3) + (age_score * 0.2)


def enrich_with_companies_house(shortlist: list[dict]) -> list[dict]:
   
    headers = _companies_house_auth_header()

    for c in shortlist:
        company_name = c.get("name", "")
        print(f"  [Stage 3] Looking up: {company_name}")

        try:
            resp = requests.get(
                "https://api.company-information.service.gov.uk/search/companies",
                headers=headers,
                params={"q": company_name, "items_per_page": 10},
                timeout=10,
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])

            active_items = [item for item in items if item.get("company_status") == "active"]

            if active_items:
                scored = [
                    (item, _score_candidate(company_name, item))
                    for item in active_items
                ]
                scored.sort(key=lambda pair: pair[1], reverse=True)
                best_match, best_score = scored[0]

                c["companies_house_number"] = best_match.get("company_number")
                c["registered_address"] = best_match.get("address_snippet")
                c["company_status"] = best_match.get("company_status")
                c["incorporation_date"] = best_match.get("date_of_creation")
                c["match_confidence"] = round(best_score, 2)

                if best_score < 0.4:
                    c["enrichment_note"] = (
                        f"Low-confidence match (score {round(best_score, 2)}):"
                        "recommend manual verification"
                    )
                else:
                    c["enrichment_note"] = ""
            else:
                c["companies_house_number"] = None
                c["match_confidence"] = 0.0
                c["enrichment_note"] = (
                    "No ACTIVE Companies House match found for this name"
                    if items
                    else "No Companies House match found for this name"
                )

        except requests.RequestException as e:
            c["companies_house_number"] = None
            c["match_confidence"] = 0.0
            c["enrichment_note"] = f"Companies House lookup failed: {e}"

        time.sleep(0.3)

    return shortlist




def run_competitor_analysis(target_product: str) -> list[dict]:
    print(f"\nRunning competitor analysis for: {target_product}\n")

    raw_results = discover_candidates(target_product)
    candidates = structure_candidates(raw_results, target_product)
    shortlist = shortlist_candidates(candidates)
    enriched = enrich_with_companies_house(shortlist)

    final_output = []
    for c in enriched:
        final_output.append(
            {
                "name": c.get("name"),
                "product_summary": c.get("product_summary"),
                "price_tier": c.get("price_tier", "Unknown"),
                "specialization": c.get("specialization", "Unknown"),
                "uk_relevance_evidence": c.get("uk_relevance_evidence"),
                "source_url": c.get("source_url"),
                "companies_house_number": c.get("companies_house_number"),
                "registered_address": c.get("registered_address"),
                "company_status": c.get("company_status"),
                "incorporation_date": c.get("incorporation_date"),
                "match_confidence": c.get("match_confidence", 0.0),
                "enrichment_note": c.get("enrichment_note", ""),
            }
        )

    output_path = OUTPUT_DIR / "competitor_profiles.json"
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nDone. Saved {len(final_output)} competitor profiles to {output_path}\n")
    return final_output


if __name__ == "__main__":
    target_product = "Dell PowerEdge enterprise servers"  # we can change this to test other products
    run_competitor_analysis(target_product)