# German Text → DGS Gloss: RNN / GRU / LSTM / Transformer

A runnable controlled comparison on **real PHOENIX14T annotations**. No video processing.
The delivered evidence is a **CPU pilot**: width 128, ten epochs, seeds 42/43/44.
The 256-width RTX 4050 configuration is provided separately; do not present pilot scores as its results.

Read `paper_notes/results_and_discussion.md` and `SUMMARY_MN.md` for measured outcomes.
The full source, downloaded annotation files, vocabularies, logs, best model weights,
predictions, metric signatures and figures are included. Raw/derived corpus data retain
RWTH's attribution/noncommercial/share-alike conditions; see `data/README.md`.

## Install (Python 3.10–3.12; tested here with Python 3.12 and CPU PyTorch 2.5.1)

Open a terminal in the extracted `text_to_gloss_comparison` folder.

Linux / Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

If PowerShell activation is restricted, use `.\.venv\Scripts\python.exe` instead of
`python` in every command; no system execution-policy change is necessary.
CUDA 12.1 wheels are from the official PyTorch 2.5.1 installation instructions.
An NVIDIA driver compatible with those wheels is needed. The local CUDA toolkit
is not required for ordinary prebuilt-wheel use. Verify the device:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

For CPU only, replace the PyTorch index above with `https://download.pytorch.org/whl/cpu`.
CUDA execution was **not** tested in this CPU-only workspace.

## Full RTX 4050 experiment: one command

Annotations are included. If supplying your own official data, put the three named
CSVs under `data/raw/` as described in `data/README.md`.

```bash
python scripts/run_all.py --prepare --config config/default.yaml --seeds 42 43 44
```

This preprocesses once, trains all four models for each seed, evaluates the held-out
test set after all training, and generates figures. The default config is width 256,
batch 16, 40-epoch maximum, validation-BLEU early stopping, seed 42, CUDA AMP when
available. For the shortest run omit `--seeds 42 43 44` to use only 42.
Training outputs go to `runs/phoenix14t_full/` and are never mixed with the pilot.

To reproduce the delivered CPU pilot into a fresh output directory:

```bash
python scripts/run_all.py --config config/reproduce_cpu.yaml --seeds 42 43 44
```

Existing checkpoint directories are protected from overwrite. Set `output` to a
new directory in the YAML for another experiment. Resume-training is not implemented;
best checkpoints are portable inference weights with config/vocabulary metadata.

## Individual commands

```bash
python -m src.data.preprocess
python -m src.train --model rnn
python -m src.train --model gru
python -m src.train --model lstm
python -m src.train --model transformer
python -m src.evaluate --model all
python -m src.figures
```

Use `--config config/cpu_pilot.yaml` to evaluate or inspect the included pilot weights.
For all included seeds:

```bash
python -m src.evaluate --config config/cpu_pilot.yaml --model all --seeds 42 43 44
python -m src.inference --config config/cpu_pilot.yaml --model gru --text "morgen ist es im norden bewölkt ."
```

Only inspect the test predictions after freezing a comparison protocol. A new
hyperparameter experiment uses development feedback, not repeated test optimization.

## Memory management

The default models are modest, but no RTX 4050 VRAM measurement is claimed.
Run `python scripts/profile_memory.py` to measure one local batch and decode pass.
This probe does not bound the full dataset's worst-case memory. Full training records
peak **allocated** GPU memory across the run in `summary.json`.
If out of memory, change `batch_size: 8` and `grad_accumulation: 2` for every model,
keeping effective batch 16. If needed use width/hidden 128 and feed-forward 256 for
all models, recording it as a separate experiment. `num_workers: 0` is portable.
A parameter-only FP32 Adam training-state lower bound is roughly 16 bytes per parameter
(weights + gradients + two moment buffers); activations, workspaces, allocator and
CUDA context are additional. This is not an estimate of total VRAM.

## Evidence and code map

| Path | Contents |
|---|---|
| `src/data/` | strict importer, train-only vocabulary, batches |
| `src/models/seq2seq.py` | shared recurrent attention implementation and Transformer |
| `src/train.py` | training, validation, best checkpoints, timings |
| `src/evaluate.py` | metrics, length groups, duplicate sensitivity, error candidates |
| `src/metrics.py` | SacreBLEU BLEU-1/2/3/4, chrF, TER and standard edit distance |
| `src/inference.py` | portable checkpoint loading and greedy decoding |
| `src/figures.py` | 300-dpi PNG plus PDF and SVG figures |
| `data/processed/phoenix14t/` | JSONL, vocabularies, exact statistics and preprocessing |
| `runs/phoenix14t_cpu_pilot/*_seed*/` | best.pt, history, config, environment and summary |
| `runs/phoenix14t_cpu_pilot/results/` | metrics.csv, predictions, length/error/seed analyses |
| `runs/phoenix14t_cpu_pilot/figures/` | eight requested figure families |
| `results/` and `figures/` | convenience copies of the delivered pilot outputs |
| `paper_notes/` | methodology, references, result/discussion sections, observations |
| `verification.txt` | actual automated-test results |

The three recurrent variants share one class to avoid accidental implementation
differences; the `kind` argument changes only the cell class. There is no requirement
for four duplicated model files.

## Verification

```bash
python -m pytest -q
```

Tests cover all four model imports, shape/forward/backward/finite gradients,
padding invariance, Transformer/recurrent causal behavior, forced-EOS stop,
checkpoint roundtrip, tiny overfit with exact greedy reconstruction, standard metric
identities, reserved vocab behavior, train-only vocabulary fitting, overlap detection,
strict split count enforcement and recurrent scheduled sampling.
No test results from CUDA are implied by the CPU tests.

## Interpretation

Do not equate a gloss sequence with a complete signed utterance. The model comparison
is matched-width, not matched-parameter. Scores concern a single weather-domain corpus
and the specified tokenization/annotation variant. Greedy decoding is identical for
all models; no beam search or invented DGS grammar baseline is included. See research
notes for a consented, linguistically validated future Mongolian corpus plan.
