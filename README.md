# CSAT Safety Gateway — Student Starter Kit

This is a **shared runner + student-defined internal system** template for the NLP team project. Every team submits Python source code; the instructor runs every submission using the **same** input/output contract and private test set. **The runner is not the student's architecture.**

## 1. What stays fixed?

- `interface.py`: defines the exact `Question` input shape.
- `runner.py`: instructor-owned common entry point. Use CLI arguments for one question or JSON/JSONL files for batch evaluation. Stdin/stdout streaming is also supported.
- For each question, it calls `student_solution.StudentSystem.respond(question: Question) -> str`.
- The return value must be a **nonempty natural-language string**. No `SOLVE`/`REFUSE` labels are required.
- One question is given to each `respond()` call; in batch mode the same `StudentSystem` object is reused throughout the process. The starter uses **exactly one FactChat API call for each question**, whether text or image.
- All debug logs go to **stderr**. `stdout` is reserved for the response protocol.

### Single-question CLI (easy local development)

```bash
# Direct text input (quote the text)
python runner.py --text "회계감사에서 중요성(materiality)이란 무엇인가요?"

# Long or multi-line text from a UTF-8 file
python runner.py --text-file questions/example.txt

# Image input from a local file
python runner.py --image questions/example.png
```

In these **single-question modes**, the program prints **only the student's unrestricted natural-language answer** to stdout unless `--output` is supplied. File output is selected by suffix:

```bash
# One question as plain natural-language text
python runner.py --text "대한민국의 수도는?" --output answer.txt

# One question as a single JSONL record: {"response": "..."}
python runner.py --text "대한민국의 수도는?" --output answer.jsonl

# An image question saved as JSONL
python runner.py --image questions/example.png --output answer.jsonl
```

Errors go to stderr and return a nonzero exit code. The runner itself handles CLI parsing; students implement only `StudentSystem.respond(question)`.
**Important:** `--output results.jsonl` on its own is not an input: specify `--input` for a batch, `--text`/`--image` for one question, or `--jsonl` for stdin streaming. `--output jsonl` is a filename literally named `jsonl`, not a file format option; include the `.jsonl` extension.

### Batch grading input (one JSON object per line)

Text:

```json
{"type":"text","text":"문제 본문 ..."}
```

Image (PNG/JPEG/WebP; path must exist in the runtime environment):

```json
{"type":"image","image_path":"/absolute/path/question.png"}
```

The question is text **or** image, not both. Image questions are given as a path to an actual local file. Paths are resolved relative to the runner's working directory if not absolute. The instructor will mount images into the evaluation environment and pass an accessible path.

### Four sample text questions (practice only)

The `examples/questions_4.jsonl` file contains **four original practice questions**:
1. CSAT-style Korean reading, 2. CSAT-style mathematics,
3. CPA-style financial accounting, 4. tax accountant-style corporate tax.
These are **newly written examples, not actual past examination questions**.

```bash
# Easiest: no < or > required. Results auto-save to results/questions_4_results.jsonl
python runner.py --input examples/questions_4.jsonl

# JSON array (.json) is accepted too
python runner.py --input examples/questions_4.json

# Set any destination explicitly
python runner.py --input examples/questions_4.jsonl --output my_results.jsonl
```

Both the `.jsonl` file (one JSON object per line) and `.json` file (an array of
question objects) are accepted. The runner uses the same `Question` class and
`StudentSystem.respond` for both. If `--output` is omitted with `--input`, a
`results/` directory is created and `{input_stem}_results.jsonl` is saved
there automatically. Existing output files with the same name are overwritten.

Individual questions can also be saved without shell redirection:

```bash
python runner.py --text "대한민국의 수도는?" --output results/answer.txt
python runner.py --image questions/example.png --output results/image_answer.txt
```

Old pipe/redirection syntax still works if desired:

```bash
python runner.py --jsonl < examples/questions_4.jsonl > results.jsonl
```

### Batch grading output (one JSON object per line)

```json
{"response":"자유로운 자연어 답변"}
```

The outer JSON object is only a machine-readable **transport envelope**. The inner `response` is unconstrained natural language and will be judged by the instructor's private LLM judge. Invalid/empty/error responses are graded as **failures**, regardless of the input category.

## 2. What may students change?

Edit `student_solution.py` freely and create additional modules. You may change prompts, build a local BERT classifier, use OCR, train models, use retrieval and search, keep a local corpus, use rules, but the submitted system must use no more than one FactChat GPT-5.4-mini API call per input question. The starter example uses **one FactChat GPT-5.4-mini API call per question**. The provided `student_solution.py` uses the exact single-prompt baseline given in the assignment. The model produces the final natural-language response directly, with no separate `BLOCK`/`ALLOW` call.

Only the course-authorized **GPT-5.4-mini via FactChat** may be used as a paid LLM/API model. Other **free/open-source models** and **external data/search** are allowed subject to applicable licenses. The FactChat helper is an optional example; the grader does not import its internals. Do not rely on another paid model or paid model API.

Keep the entry point:

```python
class StudentSystem:
    def __init__(self): ...
    def respond(self, question: Question) -> str: ...
```

Heavy model loading should happen in `__init__`, not once per question. A response must be provided for every input, and each call should handle the question independently (do not depend on test-case ordering).

## 3. Install / run

Use Python **3.11+** (subject to the instructor's final evaluation environment):

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export FACTCHAT_API_KEY="YOUR_TEAM_KEY"
python runner.py --input examples/inputs.jsonl
```

Image input example:

```bash
python runner.py --image /absolute/path/to/question.png
```

Run **offline tests without an API key**:

```bash
python -m unittest discover -s tests -v
```

## 4. FactChat gateway configuration

`factchat.py` uses the `openai` Python SDK with:

```python
OpenAI(
    api_key=os.environ["FACTCHAT_API_KEY"],
    base_url="https://factchat-cloud.mindlogic.ai/v1/gateway",
)
```

The example uses `gpt-5.4-mini`, which is listed in the public FactChat model catalog. Access still depends on the team's organization permissions; the instructor should confirm this model is enabled before distribution. The model list can be checked via `client.models.list()` or `/v1/gateway/models/`; do **not** silently substitute another model. Image content is sent in OpenAI-compatible `image_url` data-URL form; instructor should verify that the authorized model supports image input, otherwise students can implement OCR with a free/open-source vision model.

Never put FactChat API keys in source control or presentation materials. The instructor should inject keys as environment variables.

## 5. Submission requirements (interface-specific)

- All required source files, `requirements.txt`, and reproducibility instructions in the team's README.
- Any local model weights, checkpoints, and data needed for execution, or instructions/URLs to obtain them.
- No hardcoded private evaluation-set information and no paid model calls other than authorized FactChat GPT-5.4-mini.
- A `StudentSystem` class importable from `student_solution.py` and a `respond()` method returning natural language.

The full project also requires a report and presentation, to be specified in the assignment description.

## 6. Instructor deployment notes (not the private grading algorithm)

- Use one **standardized evaluation server** and a single fixed `runner.py`/`interface.py` version for all teams.
- To accommodate different packages and model weights fairly, create a **separate isolated virtualenv or container per team**, on the same standardized server infrastructure. The virtualenv/container is *not* a separate student-hosted API service.
- Set identical, published CPU/GPU/memory/time/network policies. Permit internet if online search is allowed; document what will actually be accessible.
- Run untrusted submissions as nonprivileged isolated processes, with strict resource and network limits and no access to other teams' keys or the private test set.
- Invoke `python runner.py --input hidden_test.jsonl --output team_results.jsonl` to read/write files without shell redirection. Alternatively use `--jsonl` with stdin/stdout as before. Read diagnostics from stderr. Treat exceptions, empty response lines, timeouts, malformed output, and protocol violations as invalid. Run the independent LLM judge privately on valid natural-language responses.
- Verify the exact FactChat model ID and vision support before shipping the assignment. Prefer server-side allowlisting of the permitted paid model instead of relying only on students' compliance.

## API docs

- Gateway overview: https://docs.mindlogic.ai/docs/general/api-gateway/getting-started/overview
- Model catalog: https://docs.mindlogic.ai/docs/general/api-gateway/getting-started/models
- Chat Completions: https://docs.mindlogic.ai/docs/general/api-gateway/reference/chat-completions

## 7. Batch test with four supplied image questions

The four user-provided images are included in `examples/images/`, and the
batch input file is `examples/image_questions_4.jsonl` (also available as a
JSON array at `examples/image_questions_4.json`). Each record is exactly
one image question, and the student system is called once per record.
The starter makes one FactChat request for each image (four requests for four images).

From the project root, with `FACTCHAT_API_KEY` set:

```bash
python runner.py --input examples/image_questions_4.jsonl
```

The output is automatically saved to `results/image_questions_4_results.jsonl`,
containing four lines in the same order as the input. Explicit filename:

```bash
python runner.py --input examples/image_questions_4.jsonl --output results/my_image_answers.jsonl
```

Image order: (1) English vocabulary/mop, (2) accounting, (3) mathematics,
(4) English reading comprehension. The input records contain **no labels**.
Paths in the sample file are relative to the **project root**, so run the
command from that directory. The actual GPT-5.4-mini API requests incur
FactChat credits; the offline unit tests below mock the model calls.
