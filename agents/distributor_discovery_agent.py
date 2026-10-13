"""
DISTRIBUTOR DISCOVERY AGENT

Second specialist agent in the GTM system. Its job: given a product, find
real UK distributors/resellers who could carry it, and flag how confident
we are in each match.

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



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




def discover_distributors(target_product: str, max_results_per_query: int = 6) -> list[dict]:
    
    queries = [
        f"{target_product} UK distributors",
        f"{target_product} authorized resellers UK",
        f"IT hardware wholesalers UK enterprise servers",
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




def structure_distributors(raw_results: list[dict], target_product: str) -> list[dict]:
    
    combined_text = "\n\n".join(
        f"SOURCE: {r['url']}\nTITLE: {r['title']}\nCONTENT: {r['content'][:800]}"
        for r in raw_results
    )

    system_prompt = (
        "You are a market research assistant extracting distributor/reseller "
        "companies from raw web search text. Respond with ONLY valid JSON, no "
        "other text, no markdown code fences."
    )

    user_prompt = f"""
The target product is: "{target_product}"

From the search results below, extract a list of DISTINCT companies that appear
to be DISTRIBUTORS, RESELLERS, WHOLESALERS, or CHANNEL PARTNERS for products
like this (NOT competitors who make their own competing product). Ignore
generic listicle sites or news aggregators themselves.

For each company found, return an object with these exact fields:
- "name": company name as written
- "product_categories": short phrase on what categories of product they carry
- "region_covered": UK region(s) mentioned, or "UK-wide" if not specified, or "Unknown"
- "uk_relevance_evidence": short quote or paraphrase showing UK presence, or "" if none found
- "source_url": the URL this was found in

Return a JSON array of these objects. Merge duplicate companies across sources.

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

    print(f"  [Stage 1.5] Structured {len(candidates)} candidate distributors")
    return candidates




LOW_CONFIDENCE_THRESHOLD = 0.15  


def _relevance_score(target_product: str, candidate_text: str) -> float:
    
    if not candidate_text.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([target_product, candidate_text])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(score)
    except ValueError:
        
        return 0.0


def shortlist_distributors(candidates: list[dict], target_product: str, top_n: int = 10) -> list[dict]:
    
    for c in candidates:
        candidate_text = f"{c.get('product_categories','')} {c.get('name','')}"
        relevance = _relevance_score(target_product, candidate_text)
        c["relevance_score"] = round(relevance, 3)
        c["low_confidence"] = relevance < LOW_CONFIDENCE_THRESHOLD

        uk_evidence_len = len(c.get("uk_relevance_evidence", ""))
        c["_rank_score"] = (relevance * 0.7) + (min(uk_evidence_len / 200, 1.0) * 0.3)

    candidates.sort(key=lambda c: c["_rank_score"], reverse=True)
    shortlist = candidates[:top_n]

    low_conf_count = sum(1 for c in shortlist if c["low_confidence"])
    print(
        f"  [Stage 2] Shortlisted top {len(shortlist)} candidates by relevance "
        f"({low_conf_count} flagged as low-confidence)"
    )
    return shortlist




def _companies_house_auth_header() -> dict:
    token = base64.b64encode(f"{COMPANIES_HOUSE_API_KEY}:".encode()).decode()
    return {"Authorization": f"Basic {token}"}

#literally same info
RELEVANT_SIC_CODES = {
    "26200",  # Manufacture of computers 
    "46510",  # Wholesale of computers
    "62020",  # Computer consultancy activities
    "62090",  # Other information technology and computer service activities
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


def _score_ch_candidate(target_name: str, candidate: dict) -> float:
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
                    (item, _score_ch_candidate(company_name, item))
                    for item in active_items
                ]
                scored.sort(key=lambda pair: pair[1], reverse=True)
                best_match, best_score = scored[0]

                c["companies_house_number"] = best_match.get("company_number")
                c["registered_address"] = best_match.get("address_snippet")
                c["company_status"] = best_match.get("company_status")
                c["incorporation_date"] = best_match.get("date_of_creation")
                c["match_confidence"] = round(best_score, 2)
                c["enrichment_note"] = (
                    f"Low-confidence match (score {round(best_score, 2)})"
                    if best_score < 0.4
                    else ""
                )
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



def run_distributor_discovery(target_product: str) -> list[dict]:
    print(f"\nRunning distributor discovery for: {target_product}\n")

    raw_results = discover_distributors(target_product)
    candidates = structure_distributors(raw_results, target_product)
    shortlist = shortlist_distributors(candidates, target_product)
    enriched = enrich_with_companies_house(shortlist)

    final_output = []
    for c in enriched:
        final_output.append(
            {
                "name": c.get("name"),
                "product_categories": c.get("product_categories"),
                "region_covered": c.get("region_covered", "Unknown"),
                "relevance_score": c.get("relevance_score", 0.0),
                "low_confidence": c.get("low_confidence", True),
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

    output_path = OUTPUT_DIR / "distributor_profiles.json"
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nDone. Saved {len(final_output)} distributor profiles to {output_path}\n")
    return final_output


if __name__ == "__main__":
    target_product = "Dell PowerEdge enterprise servers"
    run_distributor_discovery(target_product)