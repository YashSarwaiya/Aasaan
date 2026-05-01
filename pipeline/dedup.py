"""Near-duplicate detection via MinHash LSH.

Users will upload near-duplicate documents (copy-pasted templates, draft +
final versions, same case across multiple visits). Without dedup, the trained
model memorizes duplicates and overfits.

We use Jaccard similarity over word shingles, computed via MinHash with LSH
banding for sub-quadratic comparison time. ~50 lines.

Usage in run.py:
    docs = load_documents(...)
    docs, dropped = deduplicate(docs, threshold=0.85)  # drop docs with Jaccard > 0.85
"""

from __future__ import annotations

import re
from typing import Any


# Tunable: words below this length are stop-word-ish noise; skip them in the
# MinHash signature. Helps avoid false positives where two docs share lots of
# "the/and/of" but say different things.
MIN_TOKEN_LEN = 3


def _tokens(text: str) -> set[str]:
    """Lowercase word tokens (length >= MIN_TOKEN_LEN), set form for Jaccard."""
    words = re.findall(r"\b[a-z][a-z0-9]+\b", text.lower())
    return {w for w in words if len(w) >= MIN_TOKEN_LEN}


def deduplicate(
    docs: list[str],
    *,
    threshold: float = 0.85,
    num_perm: int = 128,
) -> tuple[list[str], int]:
    """Drop near-duplicate documents.

    Returns (kept_docs, dropped_count). For each doc, compute MinHash, query
    LSH for any near-match. If found, skip. Otherwise insert and keep.

    threshold = Jaccard threshold (0.85 = 85% token overlap considered duplicate)
    num_perm  = MinHash permutations (more = more accurate, slower; 128 is standard)

    Lazy import: datasketch isn't a hard dependency — if missing, fall back
    to no dedup with a warning. Keeps the pipeline working even on machines
    that haven't installed the optional dep yet.
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print(
            "  ⚠️  datasketch not installed; skipping dedup. "
            "Install with: pip install datasketch",
            flush=True,
        )
        return list(docs), 0

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[str] = []
    dropped = 0
    for i, doc in enumerate(docs):
        toks = _tokens(doc)
        if len(toks) < 5:
            # Doc is too short to fingerprint reliably — keep it
            kept.append(doc)
            continue

        m = MinHash(num_perm=num_perm)
        for tok in toks:
            m.update(tok.encode("utf-8"))

        if lsh.query(m):
            dropped += 1
            continue

        lsh.insert(f"doc_{i}", m)
        kept.append(doc)

    if dropped:
        print(
            f"  🧹 deduplication: kept {len(kept)}, dropped {dropped} near-duplicates",
            flush=True,
        )
    return kept, dropped
