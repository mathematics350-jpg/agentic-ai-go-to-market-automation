"""
FASTAPI INTERFACE

It runs the Orchestrator
(all four agents, in parallel) and then Report Synthesis, back to back,
and hands back the finished report in one response.

"""

from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import markdown as md_lib

from orchestrator import run_orchestrator
from agents.report_synthesis_agent import run_report_synthesis


LAST_RESULT: Optional[dict] = None

app = FastAPI(
    title="GTM Agent System",
    description=(
        "Agentic AI system for automating Go-To-Market strategy for "
        "enterprise server launches in the UK market. Enter a product "
        "category below to run the full multi-agent research and "
        "report-generation pipeline."
    ),
    version="1.0.0",
)


class GTMRequest(BaseModel):
    product: str = Field(
        ...,
        description=(
            "The product CATEGORY being launched, not a specific existing "
            "brand name, since this represents a not-yet-launched product. "
            "e.g. 'enterprise rack servers', 'mid-range enterprise servers "
            "for UK data centres'"
        ),
        json_schema_extra={"example": "enterprise rack servers"},
    )
    report_format: str = Field(
        default="summary",
        description="'summary' for flowing prose, 'table' for a structured comparison-style report",
        json_schema_extra={"example": "summary"},
    )
    preferred_model: Optional[str] = Field(
        default=None,
        description=(
            "Leave blank to run the full 3-model comparison (evaluation mode). "
            "Set to one of 'gpt-4o', 'claude-3.5', or 'gemini-pro' to generate "
            "the report with just that one model instead."
        ),
        json_schema_extra={"example": None},
    )


LANDING_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GTM Agent System</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    max-width: 720px; margin: 0 auto; padding: 60px 24px; color: #1a1a1a;
    background: #f7f8fa;
  }
  .card {
    background: white; border-radius: 12px; padding: 40px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  h1 { margin: 0 0 6px 0; font-size: 26px; }
  .subtitle { color: #666; margin: 0 0 36px 0; font-size: 15px; line-height: 1.5; }
  label { display: block; font-weight: 600; font-size: 14px; margin-bottom: 6px; margin-top: 20px; }
  label:first-of-type { margin-top: 0; }
  .hint { font-weight: 400; color: #888; font-size: 12.5px; margin-top: 3px; }
  input[type=text], select {
    width: 100%; padding: 11px 12px; font-size: 15px;
    border: 1px solid #d5d8dc; border-radius: 8px; background: white;
  }
  input[type=text]:focus, select:focus { outline: none; border-color: #2563eb; }
  button {
    margin-top: 30px; width: 100%; padding: 14px; font-size: 15px; font-weight: 600;
    background: #2563eb; color: white; border: none; border-radius: 8px; cursor: pointer;
  }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #93b4f5; cursor: not-allowed; }
  #status { margin-top: 24px; font-size: 14px; display: none; }
  #status.running { color: #2563eb; }
  #status.error { color: #c0392b; }
  .spinner {
    display: inline-block; width: 14px; height: 14px; border: 2px solid #cddaf7;
    border-top-color: #2563eb; border-radius: 50%; animation: spin 0.8s linear infinite;
    margin-right: 8px; vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
  <div class="card">
    <h1>GTM Agent System</h1>
    <p class="subtitle">
      Agentic AI system for automating Go-To-Market strategy for enterprise
      server launches in the UK market. Enter a product category to research
      competitors, distributors, regulations, and a market-entry plan.
    </p>

    <form id="gtm-form">
      <label>
        Product Category
        <div class="hint">The category of product being launched, e.g. "enterprise rack servers"</div>
      </label>
      <input type="text" id="product" placeholder="enterprise rack servers" required>

      <label>Report Format</label>
      <select id="report_format">
        <option value="summary">Summary (prose)</option>
        <option value="table">Table (structured comparison)</option>
      </select>

      <label>
        Model
        <div class="hint">Leave as "Compare all" to run the 3-model evaluation, or pick one for a single report</div>
      </label>
      <select id="preferred_model">
        <option value="">Compare all available models</option>
        <option value="gpt-4o">GPT-4o only</option>
        <option value="claude-3.5">Claude 3.5 only</option>
        <option value="gemini-pro">Gemini Pro only</option>
      </select>

      <button type="submit" id="submit-btn">Generate GTM Report</button>
    </form>

    <div id="status"></div>
  </div>

<script>
  const form = document.getElementById('gtm-form');
  const statusEl = document.getElementById('status');
  const btn = document.getElementById('submit-btn');

  form.addEventListener('submit', async function (e) {
    e.preventDefault();

    const product = document.getElementById('product').value.trim();
    const reportFormat = document.getElementById('report_format').value;
    const preferredModelRaw = document.getElementById('preferred_model').value;
    const preferredModel = preferredModelRaw === '' ? null : preferredModelRaw;

    btn.disabled = true;
    statusEl.style.display = 'block';
    statusEl.className = 'running';
    statusEl.innerHTML = '<span class="spinner"></span>Running all four research agents in parallel, then generating the report and this usually takes 30-90 seconds...';

    try {
      const response = await fetch('/generate-gtm-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          product: product,
          report_format: reportFormat,
          preferred_model: preferredModel
        })
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        throw new Error(errorBody.detail || ('Request failed with status ' + response.status));
      }

      statusEl.innerHTML = 'Done. Opening your report...';
      window.location.href = '/report-view';

    } catch (err) {
      statusEl.className = 'error';
      statusEl.textContent = 'Error: ' + err.message;
      btn.disabled = false;
    }
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
   
    return HTMLResponse(content=LANDING_PAGE_HTML)


@app.post("/generate-gtm-report")
def generate_gtm_report(request: GTMRequest):
    """
    Runs the full pipeline: all four specialist agents (in parallel),
      then Report Synthesis turns their findings into a report.
    """
    try:
        print(f"\n[API] Received request for product: {request.product}")

        orchestrator_result = run_orchestrator(request.product)

        report = run_report_synthesis(
            target_product=request.product,
            report_format=request.report_format,
            preferred_model=request.preferred_model,
        )

        result = {
            "orchestrator_summary": {
                "documents_stored": orchestrator_result.get("chromadb_documents_stored", 0),
                "errors": orchestrator_result.get("errors", []),
                "agent_durations": orchestrator_result.get("agent_durations", {}),
            },
            "report": report,
        }

        global LAST_RESULT
        LAST_RESULT = result

        return result

    except ValueError as e:
        # e.g. an invalid preferred_model, or a missing API key for it
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")


@app.get("/report-view", response_class=HTMLResponse)
def report_view():
   
    if not LAST_RESULT:
        return HTMLResponse(
            "<h1>No report yet</h1><p>Run POST /generate-gtm-report in "
            "<a href='/docs'>/docs</a> first, then come back to this page.</p>",
            status_code=404,
        )

    report_data = LAST_RESULT["report"]
    reports = report_data.get("reports", {})

    sections_html = ""
    for model_name, report_text in reports.items():
        body_html = md_lib.markdown(report_text or "", extensions=["tables"])
        sections_html += f"<div class='model-report'><h2>Generated by: {model_name}</h2>{body_html}</div><hr/>"

    page = f"""
    <html>
    <head>
    <title>GTM Report - {report_data.get('target_product', '')}</title>
    <style>
      body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 900px;
              margin: 40px auto; padding: 0 24px; line-height: 1.65; color: #1a1a1a; }}
      h1 {{ border-bottom: 3px solid #2563eb; padding-bottom: 12px; }}
      h2 {{ color: #2563eb; margin-top: 44px; }}
      table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
      th, td {{ border: 1px solid #ddd; padding: 10px 14px; text-align: left; }}
      th {{ background: #2563eb; color: white; }}
      tr:nth-child(even) {{ background: #f7f8fa; }}
      .meta {{ color: #666; font-size: 0.92em; margin-bottom: 10px; }}
      .back-link {{ display: inline-block; margin-bottom: 24px; color: #2563eb; text-decoration: none; font-size: 14px; }}
      .back-link:hover {{ text-decoration: underline; }}
    </style>
    </head>
    <body>
      <a class="back-link" href="/">&larr; New Report</a>
      <h1>Go-To-Market Report: {report_data.get('target_product', '')}</h1>
      <div class="meta">
        Format: {report_data.get('report_format', '')} &middot;
        Generated: {report_data.get('generated_at', '')} &middot;
        Models run: {', '.join(report_data.get('models_run', []))}
      </div>
      {sections_html}
    </body>
    </html>
    """
    return HTMLResponse(content=page)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)