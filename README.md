# CSAT Safety Gateway

The gateway accepts a text question or an image and returns a natural-language response. A high-scoring local match gets a refusal. Other questions go to the course-authorized FactChat GPT-5.4-mini in one API call.

- **Text** uses multilingual E5-small and the mean cosine similarity of the three closest reference embeddings.
- **Images** use raw-image EmbeddingGemma 2 at 140 vision tokens and a fitted RBF One-Class SVM.

`models/gate.npz` contains 760 text references, 359 image support vectors, the SVM coefficients, and both calibrated cutoffs. The runner reads this file at startup and loads each encoder on its first use. The detectors run inference from the saved arrays.

## Setup

Use Python 3.12. For a new checkout, the commands below install the dependencies and fetch the pinned encoder weights.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p .models/checkpoints/e5

e5_url=https://huggingface.co/intfloat/multilingual-e5-small/resolve/614241f622f53c4eeff9890bdc4f31cfecc418b3/onnx
curl -fL "$e5_url/tokenizer.json" -o .models/checkpoints/e5/tokenizer.json
curl -fL "$e5_url/model_qint8_avx512_vnni.onnx" -o .models/checkpoints/e5/model_qint8_avx512_vnni.onnx
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "google/embeddinggemma-2", revision="f9972ac",
    local_dir=".models/checkpoints/gemma",
    allow_patterns=[
        "config.json", "model.safetensors", "processor_config.json",
        "tokenizer.json", "tokenizer_config.json",
    ],
)
PY
```

Set `FACTCHAT_API_KEY` in the environment for deferred questions. Local model loading runs offline. Use `CSAT_MODEL_DIR` to point to a different directory containing the `e5` and `gemma` subdirectories. E5 runs on CPU. The image encoder uses CUDA or Apple MPS when available, with CPU as a fallback.

## Run

```bash
python runner.py --text "다음 조건을 만족하는 함수의 값을 구하시오."
python runner.py --image examples/images/question_03_math.jpg
python runner.py --input examples/questions_4.jsonl --output results/answers.jsonl
python runner.py --jsonl < examples/inputs.jsonl
```

Batch inputs use one question per line in JSONL, or a JSON array with `--input`. Image paths are relative to the current working directory unless absolute.

```json
{"type":"text","text":"문제 본문"}
{"type":"image","image_path":"examples/images/question_03_math.jpg"}
```

Single inputs print the answer. Batch outputs contain one `{"response":"..."}` record per item. Diagnostics go to stderr.

## Retrain the local detectors

Run from the project directory after setting up both encoder checkpoints.

```bash
python -m pip install -r gateway_trainer/requirements.txt
python -m gateway_trainer.train
```

The trainer reads the sibling `text/` and `image/` directories by default. It builds E5 references and fits the raw-image One-Class SVM on 2025 CSAT questions, then calibrates both cutoffs on 2026. Text records in `csat_text.jsonl` are paired with `csat_meta.jsonl` by record order. Images are split by the year at the start of their filename and deduplicated by file content.

With CSAT data alone, the cutoffs target at least 95% CSAT recall on the calibration split. Supply `--non-csat path/to/questions.jsonl` to calibrate against a 5% maximum false refusal rate for each modality represented in that file. Use the runner's JSONL format with image paths relative to the working directory or absolute. The trainer prints the calibration counts.

The default output is `models/gate.retrained.npz`. Pass `--output models/gate.npz` to write the artifact used by the runner. Use `python -m gateway_trainer.train --help` for dataset paths, years, cutoff targets, and SVM settings.

## Verification

The saved detectors were fitted on the 2025 CSAT inputs and calibrated on the 2026 split. In the 2027 benchmark, text E5 + 3-NN caught **730/760 CSAT** with **10/339 false refusals**. Raw-image SVM caught **544/777 CSAT** with **11/218 false refusals**. These counts are for the local gate.

```bash
python -m unittest discover -s tests
```
