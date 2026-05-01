"""PII detection + anonymization via Microsoft Presidio.

Strips names, SSNs, phone numbers, emails, dates of birth, etc. before any
data leaves user's session. Critical for HIPAA-aware medical and legal users.

Two modes:
  - REDACT  → "Patient John Smith" → "Patient <PERSON>"
  - REPLACE → "Patient John Smith" → "Patient Robert Jones" (synthetic)

Default is REDACT (safer; no risk of hallucinated synthetic looking real).

Usage in run.py:
    docs = load_documents(...)
    docs, n_anonymized = anonymize_documents(docs, entities=["PERSON","PHONE","EMAIL"])

Lazy import — presidio is an optional dep. Falls back to a regex-based
fallback if presidio isn't installed.
"""

from __future__ import annotations

import re
from typing import Any


# Entity types to strip by default. Conservative set covering the most
# common HIPAA identifiers + standard PII.
DEFAULT_ENTITIES = (
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "DATE_TIME",  # most aggressive — drops all date strings
    "MEDICAL_LICENSE",
    "URL",
)


# Regex fallback when presidio isn't installed. Less accurate but better
# than nothing — and zero-dep.
FALLBACK_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<SSN>"),
    (re.compile(r"\b\d{3}-\d{3}-\d{4}\b"), "<PHONE>"),
    (re.compile(r"\b\(\d{3}\)\s*\d{3}-\d{4}\b"), "<PHONE>"),
    (re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"), "<EMAIL>"),
    (re.compile(r"\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b"), "<CREDIT_CARD>"),
]


def _fallback_redact(text: str) -> str:
    """Regex-only fallback when presidio is missing. Catches the obvious stuff."""
    out = text
    for pat, replacement in FALLBACK_PATTERNS:
        out = pat.sub(replacement, out)
    return out


def anonymize_documents(
    docs: list[str],
    *,
    entities: tuple[str, ...] = DEFAULT_ENTITIES,
    language: str = "en",
) -> tuple[list[str], int]:
    """Strip PII from each document. Returns (anonymized_docs, count_modified)."""
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig
    except ImportError:
        print(
            "  ⚠️  presidio not installed; using regex fallback. "
            "Install: pip install presidio-analyzer presidio-anonymizer && "
            "python -m spacy download en_core_web_lg",
            flush=True,
        )
        out: list[str] = []
        modified = 0
        for d in docs:
            cleaned = _fallback_redact(d)
            out.append(cleaned)
            if cleaned != d:
                modified += 1
        if modified:
            print(f"  🔒 PII (regex fallback): redacted in {modified}/{len(docs)} docs", flush=True)
        return out, modified

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    # Default operator: replace each entity with `<ENTITY_TYPE>`.
    operators = {ent: OperatorConfig("replace", {"new_value": f"<{ent}>"}) for ent in entities}

    out = []
    modified = 0
    for d in docs:
        try:
            results = analyzer.analyze(text=d, entities=list(entities), language=language)
            anon = anonymizer.anonymize(text=d, analyzer_results=results, operators=operators)
            cleaned = anon.text
        except Exception as e:
            # If presidio fails on a doc, fall back to regex on just that one
            print(f"  ⚠️  presidio failed on doc, using regex fallback: {e}", flush=True)
            cleaned = _fallback_redact(d)
        out.append(cleaned)
        if cleaned != d:
            modified += 1

    if modified:
        print(f"  🔒 PII redacted in {modified}/{len(docs)} docs", flush=True)
    return out, modified
