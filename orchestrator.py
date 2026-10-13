"""
ORCHESTRATOR

This is the piece that turns four separate scripts into one coordinated
system. we give it a product (servers) then it runs all four specialist agents AT THE
SAME TIME (not one after another), waits for all of them, then stores
everything they found into ChromaDB (shared memory), so the future Report
Synthesis Agent can query across all of it in one place.

"""

import os
import sys
import json
import time
import operator
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Optional, Annotated

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
import chromadb
from chromadb.utils import embedding_functions



PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.competitor_analysis_agent import run_competitor_analysis
from agents.distributor_discovery_agent import run_distributor_discovery
from agents.regulatory_compliance_agent import run_regulatory_compliance
from agents.business_process_agent import run_business_process



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHROMA_DIR = PROJECT_ROOT / "chroma_store"
CHROMA_DIR.mkdir(exist_ok=True)

# ChromaDB needs a way to turn text into vectors (embeddings) so it can do
# similarity search later. We use OpenAI's embedding model here since we
# already have that API key set up so no new account/key needed.
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",
)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name="gtm_agent_memory",
    embedding_function=openai_ef,
)




def merge_dicts(a: Optional[dict], b: Optional[dict]) -> dict:
    """Reducer for agent_durations: combines two dicts instead of one overwriting the other."""
    merged = dict(a) if a else {}
    merged.update(b or {})
    return merged


class OrchestratorState(TypedDict):
    target_product: str
    competitor_results: Optional[list]
    distributor_results: Optional[list]
    regulatory_results: Optional[list]
    business_process_results: Optional[dict]
    errors: Annotated[list, operator.add]              # concurrent-safe: lists get concatenated
    agent_durations: Annotated[dict, merge_dicts]        # concurrent-safe: dicts get merged
    chromadb_documents_stored: int




def competitor_node(state: OrchestratorState) -> dict:
    print("[Orchestrator] Starting Competitor Analysis Agent...")
    start = time.time()
    try:
        result = run_competitor_analysis(state["target_product"])
        elapsed = round(time.time() - start, 1)
        print(f"[Orchestrator] Competitor Analysis Agent finished in {elapsed}s")
        return {"competitor_results": result, "agent_durations": {"competitor_analysis": elapsed}}
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        print(f"[Orchestrator] Competitor Analysis Agent FAILED after {elapsed}s: {e}")
        return {
            "errors": [f"Competitor Analysis Agent failed: {e}"],
            "agent_durations": {"competitor_analysis": elapsed},
        }


def distributor_node(state: OrchestratorState) -> dict:
    print("[Orchestrator] Starting Distributor Discovery Agent...")
    start = time.time()
    try:
        result = run_distributor_discovery(state["target_product"])
        elapsed = round(time.time() - start, 1)
        print(f"[Orchestrator] Distributor Discovery Agent finished in {elapsed}s")
        return {"distributor_results": result, "agent_durations": {"distributor_discovery": elapsed}}
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        print(f"[Orchestrator] Distributor Discovery Agent FAILED after {elapsed}s: {e}")
        return {
            "errors": [f"Distributor Discovery Agent failed: {e}"],
            "agent_durations": {"distributor_discovery": elapsed},
        }


def regulatory_node(state: OrchestratorState) -> dict:
    print("[Orchestrator] Starting Regulatory & Compliance Agent...")
    start = time.time()
    try:
        result = run_regulatory_compliance(state["target_product"])
        elapsed = round(time.time() - start, 1)
        print(f"[Orchestrator] Regulatory & Compliance Agent finished in {elapsed}s")
        return {"regulatory_results": result, "agent_durations": {"regulatory_compliance": elapsed}}
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        print(f"[Orchestrator] Regulatory & Compliance Agent FAILED after {elapsed}s: {e}")
        return {
            "errors": [f"Regulatory & Compliance Agent failed: {e}"],
            "agent_durations": {"regulatory_compliance": elapsed},
        }


def business_process_node(state: OrchestratorState) -> dict:
    print("[Orchestrator] Starting Business Process Agent...")
    start = time.time()
    try:
        result = run_business_process(state["target_product"])
        elapsed = round(time.time() - start, 1)
        print(f"[Orchestrator] Business Process Agent finished in {elapsed}s")
        return {"business_process_results": result, "agent_durations": {"business_process": elapsed}}
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        print(f"[Orchestrator] Business Process Agent FAILED after {elapsed}s: {e}")
        return {
            "errors": [f"Business Process Agent failed: {e}"],
            "agent_durations": {"business_process": elapsed},
        }




def store_to_chromadb_node(state: OrchestratorState) -> dict:
    
    print("[Orchestrator] All agents finished. Storing results to ChromaDB...")
    timestamp = datetime.now().isoformat()
    target_product = state["target_product"]

    # cleared out any previous run's data for this exact product first, so
    # this run's results replaces it rather than stacking on top
    existing = collection.get(where={"product": target_product})
    if existing.get("ids"):
        collection.delete(ids=existing["ids"])
        print(f"[Orchestrator] Cleared {len(existing['ids'])} documents from a previous run for this product")

    documents, metadatas, ids = [], [], []

    for i, item in enumerate(state.get("competitor_results") or []):
        doc_id = f"competitor::{i}::{item.get('name','unknown')}::{timestamp}"
        documents.append(json.dumps(item))
        metadatas.append({
            "agent": "competitor_analysis",
            "product": target_product,
            "company_name": item.get("name", ""),
            "timestamp": timestamp,
        })
        ids.append(doc_id)

    for i, item in enumerate(state.get("distributor_results") or []):
        doc_id = f"distributor::{i}::{item.get('name','unknown')}::{timestamp}"
        documents.append(json.dumps(item))
        metadatas.append({
            "agent": "distributor_discovery",
            "product": target_product,
            "company_name": item.get("name", ""),
            "timestamp": timestamp,
        })
        ids.append(doc_id)

    for i, item in enumerate(state.get("regulatory_results") or []):
        doc_id = f"regulation::{i}::{item.get('regulation_name','unknown')}::{timestamp}"
        documents.append(json.dumps(item))
        metadatas.append({
            "agent": "regulatory_compliance",
            "product": target_product,
            "regulation_name": item.get("regulation_name", ""),
            "timestamp": timestamp,
        })
        ids.append(doc_id)

    process_map = state.get("business_process_results")
    if process_map:
        for i, stage in enumerate(process_map.get("stages", [])):
            doc_id = f"process_stage::{i}::{timestamp}"
            documents.append(json.dumps(stage))
            metadatas.append({
                "agent": "business_process",
                "product": target_product,
                "stage_name": stage.get("stage_name", ""),
                "timestamp": timestamp,
            })
            ids.append(doc_id)

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print(f"[Orchestrator] Stored {len(documents)} documents in ChromaDB")
    return {"chromadb_documents_stored": len(documents)}



builder = StateGraph(OrchestratorState)

builder.add_node("competitor_analysis", competitor_node)
builder.add_node("distributor_discovery", distributor_node)
builder.add_node("regulatory_compliance", regulatory_node)
builder.add_node("business_process", business_process_node)
builder.add_node("store_to_chromadb", store_to_chromadb_node)

# fan-out: all four start at the same time, right from start
builder.add_edge(START, "competitor_analysis")
builder.add_edge(START, "distributor_discovery")
builder.add_edge(START, "regulatory_compliance")
builder.add_edge(START, "business_process")

# fan-in: store_to_chromadb only runs once all 4 have finished
builder.add_edge("competitor_analysis", "store_to_chromadb")
builder.add_edge("distributor_discovery", "store_to_chromadb")
builder.add_edge("regulatory_compliance", "store_to_chromadb")
builder.add_edge("business_process", "store_to_chromadb")

builder.add_edge("store_to_chromadb", END)

graph = builder.compile()




def run_orchestrator(target_product: str) -> dict:
    print(f"\n{'='*60}")
    print(f"ORCHESTRATOR: Running full GTM pipeline for: {target_product}")
    print(f"{'='*60}\n")

    overall_start = time.time()

    initial_state = {
        "target_product": target_product,
        "errors": [],
        "agent_durations": {},
    }
    final_state = graph.invoke(initial_state)

    overall_elapsed = round(time.time() - overall_start, 1)
    sum_of_agent_times = round(sum(final_state.get("agent_durations", {}).values()), 1)

    print(f"\n{'='*60}")
    print(f"ORCHESTRATOR: Done in {overall_elapsed}s (wall clock)")
    print(
        f"Sum of individual agent times: {sum_of_agent_times}s: this "
        f"difference is the evidence of this genuine parallel execution "
        f"(if they'd run sequentially, wall clock would be close to "
        f"{sum_of_agent_times}s, not {overall_elapsed}s)"
    )
    print(f"Documents stored in ChromaDB: {final_state.get('chromadb_documents_stored', 0)}")
    if final_state.get("errors"):
        print(f"Errors encountered ({len(final_state['errors'])}):")
        for err in final_state["errors"]:
            print(f"  - {err}")
    print(f"{'='*60}\n")

    return final_state


if __name__ == "__main__":
    target_product = "Dell PowerEdge enterprise servers"
    run_orchestrator(target_product)