"""Real pipeline ported from HiPerGator scripts.

Mapping:
  llm.py       → ask/batch_ask/extract_json (universal_pipeline.py helpers)
  schema.py    → detect_domain + build_schema + generate_questions  (steps 1-3)
  extract.py   → batch_extract_structured                            (step 4)
  qa.py        → generate_qa + filter_clean              (step5_fix.py + filter_data.py)
  train.py     → train_lora                                          (train_v2.py)
"""

from . import llm, schema, extract, qa, train  # noqa: F401
