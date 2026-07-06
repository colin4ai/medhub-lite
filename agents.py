"""Multi-agent layer for MedHub-Lite.
Orchestrator routes queries to specialist agents: QA (RAG), Extraction, Summarization.
"""
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = None

def _get_client():
    global client
    if client is None:
        client = OpenAI()
    return client
MODEL = "gpt-4o"  # or whatever qa_system uses

ROUTER_PROMPT = """You are a medical document assistant router.
Classify the user query into exactly one route:

- "qa": a natural-language QUESTION seeking an answer, even if it asks about specific fields.
  Examples: "What medications is the patient taking?", "When did the injury occur?", "Does the patient have diabetes?"
- "extract": an explicit request to EXTRACT/PULL structured data, fields, or JSON.
  Examples: "Extract the patient's key fields", "Pull structured data from this note", "Give me the fields as JSON"
- "summarize": a request for a summary or overview of a document.
  Examples: "Summarize this note", "Give me an overview of this document"

Key distinction: questions about content → "qa". Commands to produce structured records → "extract".

Respond ONLY with JSON: {"route": "<qa|extract|summarize>", "reason": "<one sentence>"}"""

EXTRACTION_SCHEMA = """Extract these fields from the medical text as JSON:
{"patient_name": str|null, "date_of_visit": str|null, "provider": str|null,
 "chief_complaint": str|null, "diagnoses": [str], "medications": [str],
 "plan": [str]}
Use null/[] when absent. Respond ONLY with JSON."""


class AgentOrchestrator:
    def __init__(self, qa_system, vector_store):
        self.qa = qa_system
        self.store = vector_store

    def run(self, query: str, top_k: int = 5) -> dict:
        trace = []

        # 1. Route
        route_raw = self._llm(ROUTER_PROMPT, query)
        route_json = self._parse_json(route_raw, default={"route": "qa", "reason": "fallback"})
        route = route_json.get("route", "qa")
        trace.append({"step": "router", "route": route, "reason": route_json.get("reason")})

        # 2. Dispatch
        if route == "extract":
            result = self._extraction_agent(query, top_k, trace)
        elif route == "summarize":
            result = self._summarizer_agent(query, top_k, trace)
        else:
            result = self._qa_agent(query, top_k, trace)

        return {"route": route, "result": result, "trace": trace}

    # --- Specialist agents ---

    def _qa_agent(self, query, top_k, trace):
        trace.append({"step": "qa_agent", "action": "delegating to RAG pipeline"})
        return self.qa.ask_question(query, top_k=top_k)   # ← adjust to real method name

    def _extraction_agent(self, query, top_k, trace):
        chunks = self._retrieve(query, top_k)
        trace.append({"step": "extraction_agent", "chunks_retrieved": len(chunks)})
        text = "\n\n".join(chunks)
        raw = self._llm(EXTRACTION_SCHEMA, text)
        return self._parse_json(raw, default={"error": "extraction_parse_failed", "raw": raw})

    def _summarizer_agent(self, query, top_k, trace):
        chunks = self._retrieve(query, top_k)
        trace.append({"step": "summarizer_agent", "chunks_retrieved": len(chunks)})
        text = "\n\n".join(chunks)
        prompt = "Summarize this medical text in 4-6 sentences for a clinician. Be factual; do not invent details."
        return {"summary": self._llm(prompt, text)}

    # --- Helpers ---

    def _retrieve(self, query, top_k):
        results = self.store.search(query, top_k=top_k)     # ← adjust to real method name
        # adjust to your store's return shape:
        return [r["content"] if isinstance(r, dict) else str(r) for r in results]

    def _llm(self, system, user):
        resp = _get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0,
        )
        return resp.choices[0].message.content

    @staticmethod
    def _parse_json(raw, default):
        try:
            return json.loads(raw.replace("```json", "").replace("```", "").strip())
        except Exception:
            return default