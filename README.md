# Aasaan — HiPerGator pipeline

Standalone CLI that turns a messy CSV/PDF/TXT into a structured training set
and (optionally) a fine-tuned LoRA adapter. Runs on a UF HiPerGator B200 GPU
under the `doctorai` conda env.

The full flow:
1. Detect the domain (medical / legal / sales / etc.)
2. Auto-build a schema (12-18 fields)
3. Generate questions over the schema
4. Extract structured forms from each document
5. Generate training Q&A pairs
6. Filter "Not specified" / hallucinated rows
7. Fine-tune Qwen 2.5 7B with LoRA (r=16, α=32, 3 epochs)

## First-time setup

```bash
ssh y.sarwaiya@hpg.rc.ufl.edu
cd /blue/dferris/y.sarwaiya
git clone https://github.com/YashSarwaiya/Aasaan.git aasaan
cd aasaan
chmod +x run-interactive.sh
```

Make sure the `doctorai` conda env has: `torch`, `transformers==4.46.0`,
`peft==0.13.2`, `trl==0.11.4`, `datasets==3.0.1`, `accelerate==1.0.1`,
`pandas`, `tqdm`, `pypdf`. (`requirements.txt` lists exact versions if you
need to make a fresh env.)

## Run with live output streaming to your terminal

```bash
./run-interactive.sh /blue/dferris/y.sarwaiya/test/mtsample/mtsamples.csv
```

Optional 2nd / 3rd args:

```bash
./run-interactive.sh path/to/data.csv <text-column> <num-docs>
./run-interactive.sh ./contracts.pdf
./run-interactive.sh ./crm_notes.csv content 500
```

You'll see every step live: domain detection, schema building, extraction
batches, Q&A generation, filtering counts, training loss curve.

## Or submit as a background SLURM job

```bash
sbatch run.sbatch /blue/dferris/y.sarwaiya/test/mtsample/mtsamples.csv
tail -f logs/aasaan-<jobid>.out
```

## Iterate on schema, train separately

```bash
# Just data prep (no LoRA training)
python run.py --input data.csv --output ./out --skip-train

# Look at ./out/structured.json — does the schema match your data?
# If not, fix the input or column and rerun.

# When happy, just train (skips re-extraction):
python run.py --output ./out --train-only
```

## Output layout

After a full run you get:

```
output_<timestamp>/
├── domain.txt                    detected domain
├── schema.json                   12-18 auto-generated fields
├── questions.json                10 generated questions
├── structured.json               extracted forms per document
├── training_data_v2.json         raw Q&A pairs
├── training_data_clean.json      filtered (the one we trained on)
├── checkpoints/                  every 50 steps, last 3 kept
├── adapter/                      final LoRA adapter (.safetensors)
└── run_metadata.json             timing + counts
```

Load the adapter for inference with `peft.PeftModel.from_pretrained(base, adapter_dir)`.
