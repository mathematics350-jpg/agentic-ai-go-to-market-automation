"""
REPORT SYNTHESIS AGENT
========================
The final piece of the pipeline. Job: query everything the four specialist
agents stored in ChromaDB for a given product, and turn it into one
coherent, readable GTM report.

"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions



load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")   
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")          

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is required (this agent always uses GPT-4o at minimum).")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CHROMA_DIR = PROJECT_ROOT / "chroma_store"
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",
)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name="gtm_agent_memory",
    embedding_function=openai_ef,
)




def retrieve_product_data(target_product: str) -> dict:
   
    result = collection.get(where={"product": target_product})

    grouped = {
        "competitor_analysis": [],
        "distributor_discovery": [],
        "regulatory_compliance": [],
        "business_process": [],
    }

    for doc, meta in zip(result.get("documents", []), result.get("metadatas", [])):
        agent = meta.get("agent")
        if agent in grouped:
            grouped[agent].append(json.loads(doc))

    total = sum(len(v) for v in grouped.values())
    print(f"  [Retrieve] Found {total} stored documents for '{target_product}':")
    for agent, items in grouped.items():
        print(f"    - {agent}: {len(items)}")

    return grouped




def build_synthesis_prompt(target_product: str, data: dict, report_format: str) -> str:
    format_instruction = (
        "Present this as a clear, structured comparison-style report using "
        "markdown tables where appropriate (e.g. a competitor comparison "
        "table), with short prose sections linking them together."
        if report_format == "table"
        else
        "Present this as flowing, readable prose organized into clear "
        "sections with headers, suitable for a business reader not just "
        "a list dump of the raw data."
    )

    return f"""
You are producing a Go-To-Market strategy report for launching "{target_product}"
in the UK enterprise market. You have been given real research findings from
four specialist research agents below. Synthesize them into ONE coherent
report. Do not invent facts beyond what is given if a section has thin
data, say so honestly rather than filling gaps with generic claims.

Write in a formal, professional business tone, as if for a corporate
strategy document read by executives. Do not use emojis, exclamation marks,
or casual language. Do not use decorative formatting such as em-dashes for
emphasis, bold text used purely for style, or bullet points where prose
would read more naturally. Headers and markdown tables (where relevant to
the format below) are the only structural formatting to use.

{format_instruction}

Structure the report with these sections:
1. Executive Summary (a few sentences)
2. Competitive Landscape (from the competitor data)
3. Distribution Strategy (from the distributor data)
4. Regulatory Considerations (from the regulatory data explicitly note
   that these require manual verification against official sources, do not
   present them as confirmed legal fact)
5. Recommended Market-Entry Process (from the business process data,
   noting which stages were grounded in real sources vs general reasoning)
6. Key Risks & Limitations (be honest about data gaps, e.g. missing revenue
   figures, low-confidence matches, small regulation sample size)

RESEARCH FINDINGS:

COMPETITORS:
{json.dumps(data['competitor_analysis'], indent=2)}

DISTRIBUTORS:
{json.dumps(data['distributor_discovery'], indent=2)}

REGULATIONS:
{json.dumps(data['regulatory_compliance'], indent=2)}

MARKET-ENTRY PROCESS:
{json.dumps(data['business_process'], indent=2)}
"""




def _run_gpt4o(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


def _run_claude(prompt: str) -> str | None:
    """

    NOTE: model version strings get updated over time, so please don't assume the one below is still
    current whenever we get here.
    """
    if not ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic
    except ImportError:
        print("  [Models] ANTHROPIC_API_KEY is set but the 'anthropic' package isn't installed "
              "(pip install anthropic) skipping Claude for now.")
        return None

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")  
    response = client.messages.create(
        model=model_name,
        max_tokens=8192,  
        messages=[{"role": "user", "content": prompt}],
    )

    
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError("Claude response contained no text block (only thinking/other blocks)")


def _run_gemini(prompt: str) -> str | None:
    """
     NOTE: we can check aistudio.google.com for the current
    model name when we actually add this key.
    """
    if not GOOGLE_API_KEY:
        return None
    try:
        import google.generativeai as genai
    except ImportError:
        print("  [Models] GOOGLE_API_KEY is set but 'google-generativeai' isn't installed "
              "(pip install google-generativeai) skipping Gemini for now.")
        return None

    genai.configure(api_key=GOOGLE_API_KEY)
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")  
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text


MODEL_RUNNERS = {
    "gpt-4o": _run_gpt4o,
    "claude-3.5": _run_claude,
    "gemini-pro": _run_gemini,
}

MODEL_KEY_REQUIREMENTS = {
    "gpt-4o": "OPENAI_API_KEY",
    "claude-3.5": "ANTHROPIC_API_KEY",
    "gemini-pro": "GOOGLE_API_KEY",
}


def run_all_available_models(prompt: str) -> dict:
    """
    NOTE: Claude and Gemini calls are wrapped in try/except so a
    live API error from one (e.g. a deprecated model name, a rate limit) does
    not throw away GPT-4o's result
    """
    results = {}
    errors = {}

    print("  [Models] Running GPT-4o...")
    start = time.time()
    results["gpt-4o"] = _run_gpt4o(prompt)
    print(f"  [Models] GPT-4o done in {round(time.time() - start, 1)}s")

    try:
        claude_result = _run_claude(prompt)
        if claude_result:
            results["claude-3.5"] = claude_result
            print("  [Models] Claude 3.5 done")
    except Exception as e:
        errors["claude-3.5"] = str(e)
        print(f"  [Models] Claude 3.5 FAILED (skipping, keeping other results): {e}")

    try:
        gemini_result = _run_gemini(prompt)
        if gemini_result:
            results["gemini-pro"] = gemini_result
            print("  [Models] Gemini Pro done")
    except Exception as e:
        errors["gemini-pro"] = str(e)
        print(f"  [Models] Gemini Pro FAILED (skipping, keeping other results): {e}")

    if errors:
        results["_model_errors"] = errors

    return results


def run_single_model(prompt: str, preferred_model: str) -> dict:
    
    if preferred_model not in MODEL_RUNNERS:
        raise ValueError(
            f"Unknown model '{preferred_model}'. Choose one of: {list(MODEL_RUNNERS.keys())}"
        )

    required_key = MODEL_KEY_REQUIREMENTS[preferred_model]
    if not os.getenv(required_key):
        raise ValueError(
            f"Cannot use '{preferred_model}' -- {required_key} is not set in your .env file."
        )

    print(f"  [Models] Running {preferred_model} only (user-selected)...")
    start = time.time()
    report_text = MODEL_RUNNERS[preferred_model](prompt)
    print(f"  [Models] {preferred_model} done in {round(time.time() - start, 1)}s")

    return {preferred_model: report_text}




def run_report_synthesis(
    target_product: str,
    report_format: str = "summary",
    preferred_model: str | None = None,
) -> dict:
    
    print(f"\nRunning report synthesis for: {target_product} (format: {report_format})\n")

    data = retrieve_product_data(target_product)
    total_items = sum(len(v) for v in data.values())
    if total_items == 0:
        raise ValueError(
            f"No stored data found for '{target_product}'. Run orchestrator.py "
            f"for this exact product name first Report Synthesis only reads "
            f"from ChromaDB, it doesn't do its own research."
        )

    prompt = build_synthesis_prompt(target_product, data, report_format)

    if preferred_model:
        model_reports = run_single_model(prompt, preferred_model)
        model_errors = {}
    else:
        model_reports = run_all_available_models(prompt)
        model_errors = model_reports.pop("_model_errors", {})

    output = {
        "target_product": target_product,
        "report_format": report_format,
        "generated_at": datetime.now().isoformat(),
        "models_run": list(model_reports.keys()),
        "model_errors": model_errors,
        "reports": model_reports,
        "source_data_counts": {k: len(v) for k, v in data.items()},
    }

    output_path = OUTPUT_DIR / "gtm_report.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. Report saved to {output_path}")
    print(f"Models run: {output['models_run']}")
    if preferred_model is None and output["models_run"] == ["gpt-4o"]:
        print(
            "Only GPT-4o ran and this is expected until ANTHROPIC_API_KEY and "
            "GOOGLE_API_KEY are added to .env. The report itself is still complete."
        )
    print()

    return output


if __name__ == "__main__":
    target_product = "Dell PowerEdge enterprise servers"
    run_report_synthesis(target_product, report_format="summary")