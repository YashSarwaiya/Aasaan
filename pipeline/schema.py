"""Domain detection + schema generation. Ports steps 1-3 of universal_pipeline.py."""

from __future__ import annotations

import json
from typing import Any

from .llm import ask, extract_json, extract_json_list


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
DEFAULT_SCHEMA: dict[str, str] = {
    "patient_age": "patient age in years",
    "patient_sex": "M or F",
    "chief_complaint": "main reason for visit",
    "history_of_present_illness": "current illness details",
    "past_medical_history": "prior conditions",
    "medications": "current medications list",
    "allergies": "known allergies",
    "vital_signs": "blood pressure, HR, temp",
    "physical_exam": "exam findings",
    "diagnosis": "diagnosis or assessment",
    "treatment_plan": "treatment plan",
    "follow_up": "follow-up instructions",
}


def build_schema(model, tokenizer, domain: str, sample_docs: list[str]) -> dict[str, str]:
    """Step 2: ask the model to propose a 12-field schema. With retry + fallback."""
    schema_sample = "\n\n---\n\n".join(sample_docs[:15])[:8000]
    prompt = f"""Domain: {domain}

Here are 15 sample documents from this domain:
{schema_sample}

Build a JSON schema with 12 important data fields to extract from these documents.

Rules:
- Use snake_case field names
- Each field has a short description
- Output ONLY valid JSON, no markdown, no explanations
- Must have exactly 12 fields

Example format (for medical):
{{"patient_age": "age in years", "chief_complaint": "main reason for visit", "medications": "list of current meds", "diagnosis": "medical diagnosis", "vital_signs": "BP, HR, temp", "allergies": "known allergies", "past_history": "prior conditions", "physical_exam": "exam findings", "labs": "lab results", "treatment_plan": "plan going forward", "follow_up": "next steps", "specialty": "medical field"}}

Now output the JSON schema for {domain} with exactly 12 fields:"""
    response = ask(model, tokenizer, prompt, max_tokens=1500)
    schema = extract_json(response)

    if schema is None or len(schema) < 5:
        retry_prompt = f"""Output a JSON object with exactly 12 fields for {domain}.
Format: {{"field1": "desc", "field2": "desc", ..., "field12": "desc"}}
ONLY JSON, nothing else:"""
        response = ask(model, tokenizer, retry_prompt, max_tokens=1000)
        schema = extract_json(response)

    if schema is None or len(schema) < 5:
        return dict(DEFAULT_SCHEMA)
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
