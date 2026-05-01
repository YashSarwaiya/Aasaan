"""Domain detection + schema generation. Ports steps 1-3 of universal_pipeline.py.

Schema fields are classified into two groups so downstream training and serving
can handle them correctly:

  - LOOKUP fields (factual: meds, vitals, labs, dates) → served via RAG over
    the structured form. Not used as training targets — fine-tuning on lookup
    facts wastes model capacity and produces hallucinations (e.g. inventing
    drug names like "Ancef").
  - REASONING fields (synthesized: plan, assessment, differential, red_flags) →
    used as training targets in the multi-task curriculum. These benefit from
    fine-tuning because they encode clinical judgment patterns that vary by
    practice.

Classification is keyword-based with a 'default to reasoning' fallback for
ambiguous fields (since the user can always demote a field via the UI later).
"""

from __future__ import annotations

import json
from typing import Any

from .llm import ask, extract_json, extract_json_list


# Keyword groups for field classification. A field is LOOKUP if it has more
# lookup hits than reasoning hits, REASONING otherwise. Ties default to
# reasoning so we don't silently exclude a useful training signal.

LOOKUP_KEYWORDS = (
    "medication", "drug", "rx ", "prescription", "pharma",
    "vital", "blood pressure", "heart rate", "temperature", "pulse", "bp ", "hr ",
    "lab ", "laboratory", "test result", "blood work",
    "exam", "examination", "physical exam",
    "age", "sex", "gender", "ethnicity",
    "name", "_id", " id", "patient_id", "mrn",
    "date", "time", "_at", "_on",
    "allergi",
    "vitals", "labs", "weight", "height", "bmi",
)

REASONING_KEYWORDS = (
    "plan", "treatment plan", "follow up", "follow-up", "next step",
    "assessment", "impression", "summary",
    "diagnos", "differential", "indication",
    "red flag", "risk", "complication", "warning",
    "recommendation", "advice", "conclusion",
    "complaint", "chief complaint",
    "history of present", "hpi",
    "additional notes", "additional_notes", "notes",
    "reasoning", "rationale", "why",
)


def classify_fields(schema: dict[str, str]) -> tuple[list[str], list[str]]:
    """Split schema fields into (lookup_fields, reasoning_fields).

    LOOKUP fields are served via RAG. REASONING fields are training targets.
    Heuristic: count keyword hits in field name + description. Tie or no hits
    defaults to REASONING (we'd rather train on a borderline field than miss it).
    """
    lookup: list[str] = []
    reasoning: list[str] = []
    for name, desc in schema.items():
        text = (name + " " + (desc or "")).lower()
        lookup_hits = sum(1 for kw in LOOKUP_KEYWORDS if kw in text)
        reasoning_hits = sum(1 for kw in REASONING_KEYWORDS if kw in text)
        if lookup_hits > reasoning_hits:
            lookup.append(name)
        else:
            reasoning.append(name)
    return lookup, reasoning


def detect_domain(model, tokenizer, sample_docs: list[str]) -> str:
    """Step 1: ask the model to label the domain in 2-5 words.

    Uses 5 sample docs (concatenated, capped at 3000 chars).
    """
    sample_text = "\n\n---\n\n".join(sample_docs[:5])[:3000]
    prompt = f"""Look at these 5 sample documents. What domain/industry are they from?

DOCUMENTS:
{sample_text}

Output ONLY the domain name in 2-5 words.
Examples: "medical clinical notes", "legal contracts", "sales call logs", "software bug reports"
Domain:"""
    domain = ask(model, tokenizer, prompt, max_tokens=20).strip().strip('"').strip("'")
    return domain.split("\n")[0].strip()


# Fallback schema — used only when both LLM attempts produce malformed JSON.
# Includes both lookup (factual) and reasoning (synthesized) fields.
DEFAULT_SCHEMA: dict[str, str] = {
    # lookup
    "patient_age": "patient age in years",
    "patient_sex": "M or F",
    "current_medication": "current medications list",
    "allergies": "known allergies",
    "vitals": "blood pressure, HR, temp",
    "physical_examination": "exam findings",
    "laboratory_data": "lab results",
    # reasoning
    "chief_complaint": "main reason for visit",
    "history_of_present_illness": "current illness story",
    "past_medical_history": "prior conditions in narrative form",
    "assessment": "clinician's diagnostic reasoning",
    "differential_diagnosis": "alternative diagnoses considered",
    "impression_and_plan": "summary impression and treatment plan",
    "red_flags": "critical concerns to watch for",
}


def build_schema(model, tokenizer, domain: str, sample_docs: list[str]) -> dict[str, str]:
    """Step 2: ask the model to propose a ~14-field schema covering both lookup
    and reasoning content. With retry + fallback.

    The prompt explicitly requests reasoning fields (assessment, differential,
    red_flags, etc.) because earlier versions only auto-detected lookup fields,
    starving the multi-task curriculum of training signal.
    """
    schema_sample = "\n\n---\n\n".join(sample_docs[:15])[:8000]
    prompt = f"""Domain: {domain}

Here are 15 sample documents from this domain:
{schema_sample}

Build a JSON schema with 14 important data fields to extract from these documents.

Rules:
- Use snake_case field names
- Each field has a short description
- Output ONLY valid JSON, no markdown, no explanations
- Must have exactly 14 fields
- Skip identifier fields like "patient_name" / "patient_id" — anonymized data won't have them
- INCLUDE both factual fields (vitals, meds, labs) AND reasoning fields (assessment,
  differential_diagnosis, red_flags, risk_factors). Reasoning fields encode the
  practitioner's judgment and are critical for fine-tuning.

Example format (for medical clinical notes):
{{"chief_complaint": "main reason for visit", "history_of_present_illness": "current illness story", "past_medical_history": "prior conditions", "current_medication": "list of current meds", "allergies": "known allergies", "vitals": "BP, HR, temp", "physical_examination": "exam findings", "laboratory_data": "lab results", "assessment": "clinician's diagnostic reasoning", "differential_diagnosis": "alternative diagnoses considered", "impression_and_plan": "summary impression and treatment plan", "red_flags": "critical concerns to watch for", "risk_factors": "patient-specific risk factors", "follow_up": "next steps and follow-up instructions"}}

Now output the JSON schema for {domain} with exactly 14 fields, including AT LEAST FIVE reasoning fields (assessment, differential, red_flags, etc.):"""
    response = ask(model, tokenizer, prompt, max_tokens=1800)
    schema = extract_json(response)

    if schema is None or len(schema) < 5:
        retry_prompt = f"""Output a JSON object with exactly 14 fields for {domain}.
Include reasoning fields like assessment, differential_diagnosis, red_flags, risk_factors.
Skip identifier fields like patient_name.
Format: {{"field1": "desc", "field2": "desc", ..., "field14": "desc"}}
ONLY JSON, nothing else:"""
        response = ask(model, tokenizer, retry_prompt, max_tokens=1200)
        schema = extract_json(response)

    if schema is None or len(schema) < 5:
        return dict(DEFAULT_SCHEMA)

    # Strip any patient_name / patient_id field that snuck through despite the prompt.
    schema = {
        k: v for k, v in schema.items()
        if k not in ("patient_name", "patient_id", "name", "mrn")
    }
    return schema


def generate_questions(
    model, tokenizer, domain: str, schema: dict[str, str]
) -> list[str]:
    """Step 3: ask for 10 useful questions over the schema. Falls back to per-field templates."""
    prompt = f"""Domain: {domain}
Available fields: {list(schema.keys())}

Generate exactly 10 useful questions a user would ask about documents in this domain.

Output ONLY a JSON list of 10 question strings, nothing else.
Example: ["What is the diagnosis?", "List the medications.", ...]

JSON list:"""
    response = ask(model, tokenizer, prompt, max_tokens=600)
    questions = extract_json_list(response)
    if not questions or len(questions) < 5:
        questions = [
            f"What is the {field.replace('_', ' ')}?"
            for field in list(schema.keys())[:10]
        ]
    return [str(q) for q in questions[:10]]


def schema_to_typed(schema: dict[str, str]) -> list[dict[str, Any]]:
    """Convert {name: description} → [{name, type, example}] for the frontend.

    Naive type inference based on field name keywords. Frontend type editor lets
    users override these.
    """
    out: list[dict[str, Any]] = []
    for name, desc in schema.items():
        ftype = "text"
        lower = (name + " " + desc).lower()
        if any(k in lower for k in ("date", "_at", "_on")):
            ftype = "date"
        elif any(k in lower for k in ("amount", "size", "count", "age", "_id", "qty")):
            ftype = "number"
        elif any(k in lower for k in ("list", "items", "medications", "tags")):
            ftype = "list"
        out.append({"name": name, "type": ftype, "example": str(desc)[:80]})
    return out


def typed_to_schema(typed: list[dict[str, Any]]) -> dict[str, str]:
    """Inverse of schema_to_typed — what train_pipeline gets from the frontend."""
    return {f["name"]: str(f.get("example") or f["name"]) for f in typed if f.get("name")}
