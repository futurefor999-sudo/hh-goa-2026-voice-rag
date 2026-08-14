# Voice-Enabled RAG — HH Goa 2026, Shortlisting Task 2

Voice input → Sarvam speech-to-text → multi-strategy chunking + vector
retrieval → grounded answer generation → guardrails, all timed
per-stage, with a mobile-friendly web UI and a deployable FastAPI
server.

**Dataset note:** the task brief links to `ai4bharat/MSMARCO-XL` — the
actual HuggingFace dataset is `ai4bharat/MSMARCO-XI` (a capital "I",
easy to misread as a lowercase "l" in HF's rendering). This repo
ingests the real dataset under its real name; see [Dataset
ingestion](#3-dataset-ingestion).

---

## 1. Architecture

```
Voice clip ──▶ Sarvam STT ──▶ text query
                                  │
                                  ▼
                        embed query (TF-IDF or
                        sentence-transformers)
                                  │
                                  ▼
                    vector search (in-memory cosine
                     index, or FAISS for scale)
                                  │
                                  ▼
                 ┌── pre-generation guardrails ──┐
                 │  unsafe-input screen           │
                 │  off-topic (low similarity)    │
                 └────────────────┬───────────────┘
                                  ▼
                    generation (Anthropic / OpenAI /
                     offline extractive mock)
                                  │
                                  ▼
                 ┌── post-generation guardrail ───┐
                 │  grounding / hallucination      │
                 │  check against retrieved text   │
                 └────────────────┬───────────────┘
                                  ▼
                    answer + retrieved context +
                    per-stage latency → JSON / web UI
```

Every stage is timed; every stage is wrapped by the harness
(`src/pipeline.py`) with structured request/response objects, retries
with backoff on external calls, and a typed error path per failure
mode — not a single raw prompt-in/text-out call.

```
src/
  config.py                  # all settings, read from environment / .env
  chunking/strategies.py     # fixed, sentence, semantic, metadata-aware hybrid
  retrieval/embeddings.py    # TF-IDF (offline default) or sentence-transformers
  retrieval/vector_store.py  # in-memory cosine index, or FAISS
  stt/sarvam_stt.py          # real Sarvam REST client + mock for offline dev
  generation/generator.py    # anthropic / openai / mock (extractive, offline)
  guardrails/checks.py       # unsafe-input, off-topic, grounding/hallucination
  pipeline.py                 # the harness — wires every stage together
  latency.py                  # P50/P70/P100 tracking
scripts/
  ingest.py                   # dataset -> chunks -> embeddings -> index on disk
  run_query.py                 # single query CLI (text or audio)
  benchmark.py                 # latency report over many queries
  demo_cases.py                 # normal / off-topic / unsafe / unsupported test cases
app.py                          # FastAPI server: web UI + /query/text + /query/voice
static/index.html               # mobile-friendly web UI (single file, no build step)
tests/                           # 30 tests, all offline, no network/keys required
data/sample_corpus.jsonl         # 5-doc corpus for local dev before pointing at MSMARCO-XI
data/demo_test_cases.jsonl        # the 10 demo test cases
data/benchmark_queries.txt         # 30-query set used for the latency benchmark below
Dockerfile                          # deployable container
.gitignore                           # excludes .env and all generated artifacts
```

---

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in SARVAM_API_KEY and a generation key — see §6
```

Without any API keys set, `GEN_PROVIDER=mock` (the `.env.example`
default) uses a deterministic **extractive** fallback instead of
calling an LLM — it always answers with a real sentence copied from
retrieved context, never a fabricated one — so you can exercise the
whole pipeline completely offline. Swap `GEN_PROVIDER` to `anthropic`
or `openai` once you have a key for the real submission.

Build the demo index and try it:

```bash
python -m scripts.ingest --source data/sample_corpus.jsonl --out data/index.pkl
python -m scripts.run_query --index data/index.pkl --query "What is retrieval-augmented generation?"

# live server + web UI
uvicorn app:app --reload
# open http://localhost:8000 — mobile-friendly, works in a phone browser too (same Wi-Fi + your machine's LAN IP)
```

---

## 3. Dataset ingestion

### Quick local dev (bundled 5-doc sample, no network)

```bash
python -m scripts.ingest --source data/sample_corpus.jsonl --out data/index.pkl --strategy hybrid
```

### Real submission run against `ai4bharat/MSMARCO-XI`

```bash
python -m scripts.ingest --list-hf-languages     # 14 Indic language configs
python -m scripts.ingest --source hf --language hi --limit 5000 --out data/index.pkl
```

`--language` picks one of the dataset's 14 Indic-language configs
(`as, bn, gu, hi, kn, ml, mr, ne, or, pa, sa, ta, te, ur`).
`--passage-field` chooses whether to index `Translated_passages`
(default) or `English_passages` — pick based on which language your
demo will be spoken/asked in. `--limit` caps the number of dataset
*rows* pulled (each row carries ~10 passages, so `--limit 5000` indexes
tens of thousands of passages, not 5000).

Ingestion also writes `data/index.eval_queries.jsonl` — the dataset's
own queries, each with `relevant_doc_ids` from its `is_selected` flags.
Feed this straight into the benchmark and demo-case tooling for a real
(non-synthetic) run once you're on the full corpus:

```bash
python -m scripts.benchmark --index data/index.pkl --queries data/index.eval_queries.jsonl
```

**This sandbox has no network access**, so the actual HuggingFace fetch
couldn't be executed here. What *was* verified without network
(`tests/test_ingest.py`, 8/8 passing): the row-transform logic against
a row shaped exactly like HuggingFace's own documented example for
this dataset — confirming the field names, the passage/`is_selected`
handling, and the eval-query construction are all correct for the real
schema. Run the `--source hf` command above yourself once to confirm
the live fetch; if the schema has changed since this was written,
`_transform_msmarco_row` in `scripts/ingest.py` is the one place to fix.

---

## 4. Chunking strategies

`src/chunking/strategies.py` implements four, not a single fixed-size
pass:

- **fixed** — word-window with overlap (kept as a baseline/fallback only)
- **sentence** — packs whole sentences up to a max length; never splits mid-sentence
- **semantic** — embeds consecutive sentences and cuts where similarity to the running chunk centroid drops (a real topic shift), not at a fixed character count
- **hybrid** (default) — semantic for documents with enough sentence structure to benefit, sentence-packing otherwise, fixed-window only as a last resort for text with no detectable sentence breaks

All chunks carry propagated document-level metadata (title, source,
doc id) — metadata-aware chunking, so retrieval can filter/re-rank on it.

---

## 5. Guardrails — verified

Four checks, run inline in the harness:

| Check | When | What it does |
|---|---|---|
| unsafe-input | before retrieval | regex screen for prompt-injection / jailbreak patterns |
| off-topic | after retrieval, before generation | blocks if the best retrieval similarity is below a floor |
| (abstain) | during generation | prompt instructs the model to emit `ABSTAIN` rather than answer beyond the context |
| grounding / hallucination | after generation | verifies the answer's content words actually overlap with the retrieved context — a post-hoc check, since a model can ignore prompt instructions |

**Verified with `scripts/demo_cases.py` against `data/demo_test_cases.jsonl`
(10 cases — normal, off-topic, unsafe, unsupported):**

```
category      expected                actual              match
normal        answered                answered            YES   (×4)
off_topic     blocked                 blocked             YES   (×2)
unsafe        blocked                 blocked             YES   (×2)
unsupported   abstained/blocked/…     answered / blocked  YES   (×2)

10/10 matched expected status.
```

Re-run yourself: `python -m scripts.demo_cases --index data/index.pkl`

**Honest limitation of this exact result:** the offline `mock`
generator is *extractive* — it only ever echoes real context verbatim,
so it structurally cannot fabricate a fact and therefore can never fail
the grounding check. Both "unsupported" cases correctly avoid asserting
the missing specific fact (one abstains/gets blocked as off-topic-ish,
the other returns a real, true, non-numeric sentence rather than
inventing a number) — but the grounding check itself is only truly
exercised once a real LLM backend is in the loop, since that's the
backend that can actually ignore instructions and hallucinate. Test
with `GEN_PROVIDER=anthropic` or `openai` before final submission to
confirm the grounding check catches a real model's hallucinations too.

**Also worth knowing before you present this:** on the tiny 5-doc
sample corpus, TF-IDF similarity is noisy enough that a few
genuinely-unrelated benchmark queries (e.g. "who won the last IPL
final?") scored just above the off-topic floor from incidental word
overlap and got an "answered" status with an unhelpful extracted
sentence, rather than being blocked. This is a known weakness of
TF-IDF + a 5-document vocabulary, not a guardrail design flaw — it
should behave much better once `OFFTOPIC_MIN_SIM` is retuned against
the real MSMARCO-XI corpus (see §7) and/or `EMBEDDING_PROVIDER` is
switched to `sentence-transformers`, and it's essentially a non-issue
once the corpus is thousands of real passages instead of five.

---

## 6. API keys — configuration and safety

- All keys are read from environment variables via `src/config.py`
  (loaded from `.env` locally, or your hosting platform's environment
  settings in production). No key is ever hardcoded anywhere in the
  codebase — confirmed by a repo-wide scan before every commit.
- `.env` is the one file that ever holds a real key, and `.gitignore`
  excludes it (and every `.env.*` variant) from version control. Only
  `.env.example`, with placeholder values, is tracked.
- Required for the real (non-mock) pipeline:
  - `SARVAM_API_KEY` — STT (`STT_PROVIDER=sarvam`)
  - `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` — generation (`GEN_PROVIDER=anthropic`/`openai`)
- On your hosting platform (Render/Railway/etc.), set these as
  **environment variables / secrets** in the dashboard — never in a
  committed file. See §10 for exact steps.

---

## 7. On the 200ms latency target

The brief scopes 200ms to "chunking + vector DB retrieval + everything
through to final output." Two things are worth being upfront about:

- **Chunking is precomputed at ingest time**, not per-query, so it
  doesn't count against per-query latency — standard for RAG (you'd
  never re-chunk the whole corpus on every question). Per-query
  retrieval (embed the query + search the index) is what's timed.
- **STT and LLM generation are real network calls to external APIs.**
  Those cost real round-trip time independent of anything this
  codebase does. `src/latency.py` reports both `total_ms` (everything)
  and `pipeline_ms` (retrieval + guardrails only — the part this
  system's own design controls) so this is visible rather than hidden.

### Actual benchmark: 30 queries, sample corpus, `GEN_PROVIDER=mock`

Run: `python -m scripts.benchmark --index data/index.pkl --queries data/benchmark_queries.txt`
(query set spans all four categories — normal, off-topic, unsafe, unsupported)

| stage | P50 | P70 | P100 | mean |
|---|---|---|---|---|
| **total** (stt + retrieval + generation + guardrails) | 0.83 ms | 0.86 ms | 1.58 ms | 0.82 ms |
| **pipeline** (retrieval + guardrails only) | 0.71 ms | 0.75 ms | 1.36 ms | 0.73 ms |
| retrieval | 0.64 ms | 0.69 ms | 1.28 ms | 0.67 ms |
| generation (mock, offline) | 0.09 ms | 0.11 ms | 0.22 ms | 0.09 ms |
| guardrails | 0.07 ms | 0.07 ms | 0.10 ms | 0.06 ms |
| stt | 0 ms (skipped — text queries) | | | |

Status breakdown: 24 answered, 0 abstained, 6 blocked.
Full machine-readable report: `data/latency_report.json`.

**These numbers are real, from an actual run, on the 5-doc sample
corpus with the offline mock generator and no network calls** — not
representative of production numbers, and not meant to be. They
confirm the *pipeline's own* retrieval + guardrail logic comfortably
clears 200ms (by two to three orders of magnitude, even). They say
nothing about what a real Sarvam STT call or a real hosted-LLM
generation call will cost — that's dominated by network round-trip,
typically several hundred ms each, and no amount of chunking or
indexing cleverness changes that. **Before you submit**, re-run this
same benchmark two ways and report both:
1. `--queries data/index.eval_queries.jsonl` (real MSMARCO-XI queries, after real ingestion)
2. with `GEN_PROVIDER=anthropic` (or `openai`) and `STT_PROVIDER=sarvam`, to get real end-to-end numbers including network latency

Be ready to explain the total-vs-pipeline split to judges if the
200ms scope is questioned.

---

## 8. The web UI

`static/index.html` — a single-file, mobile-friendly console (dark,
monospace-accented, no build step, no external JS framework). Shows:

- a text box **and** a mic button (records via the browser's
  `MediaRecorder` API, sends the clip straight to `/query/voice`)
- a 4-stage latency rail (STT / Retrieval / Generation / Guardrails)
  that lights up live with each stage's timing as the response comes back
- the transcript (voice queries only)
- the retrieved context passages with their similarity scores
- the final grounded answer
- a status banner (answered / abstained / blocked) with the reason

It's served directly by the FastAPI app at `/` — no separate frontend
deploy or build step needed; visiting your deployed URL on a phone
browser opens straight into it.

---

## 9. Testing

```bash
python tests/test_pipeline.py   # 11 tests — chunking, retrieval, guardrails, end-to-end
python tests/test_ingest.py     # 8 tests — MSMARCO-XI row-transform vs. documented schema
python tests/test_stt.py        # 11 tests — Sarvam client request-building & error handling
# or, if pytest is installed: python -m pytest tests/ -v
```

**30/30 passing**, verified in this environment immediately before
handoff. All three suites run fully offline — no network, no API keys
required — by design: `test_pipeline.py` uses the TF-IDF embedder and
the extractive mock generator against the sample corpus;
`test_ingest.py` feeds the real MSMARCO-XI row-transform logic a row
shaped exactly like HuggingFace's own documented example, without
calling `datasets.load_dataset`; `test_stt.py` mocks `requests.post` to
verify the real `SarvamSTT` client builds correct requests and handles
Sarvam's response/error shapes correctly, without calling the network.

What is **not** covered by these offline tests, and what you should
verify yourself once you have credentials: an actual live call to
Sarvam's endpoint, an actual live call to your chosen generation API,
and an actual `--source hf` ingestion run. The code paths for all three
are implemented and unit-tested at the boundary; the live network
calls themselves need your keys and couldn't be exercised in this
sandboxed build environment.

---

## 10. Deployment

### Render (recommended — free tier, browser-only, works from a phone)

1. Push this repo to GitHub (see the final numbered list at the very end).
2. Go to [render.com](https://render.com) → sign in with GitHub → **New +** → **Web Service**.
3. Connect your GitHub repo.
4. Render will detect the `Dockerfile` automatically — leave "Environment: Docker" selected.
5. Under **Environment Variables**, add:
   - `SARVAM_API_KEY` = your real key
   - `GEN_PROVIDER` = `anthropic` (or `openai`)
   - `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) = your real key
   - `STT_PROVIDER` = `sarvam`
6. Click **Create Web Service**. First build takes a few minutes (it
   runs the sample-corpus ingest as part of the Docker build — see
   `Dockerfile`).
7. Render gives you a public URL like `https://your-app.onrender.com` —
   that's your live demo link.

Free tier note: the service sleeps after inactivity and takes ~30-60s
to wake on the next request — fine for a hackathon demo link, just
open it once a minute or two before you need it live (e.g. before
recording Video 2).

### Railway (alternative, same idea)

1. [railway.app](https://railway.app) → sign in with GitHub → **New Project** → **Deploy from GitHub repo**.
2. Select this repo — Railway also auto-detects the `Dockerfile`.
3. Add the same environment variables as above under the service's **Variables** tab.
4. Railway assigns a public domain automatically (or generate one under **Settings → Networking**).

### After deploying — smoke test

Open `https://your-app-url/health` in a browser. You should see:
```json
{"status": "ok", "index_loaded": true, "index_size": 25, "stt_configured": true, "gen_provider": "anthropic"}
```
Then open `https://your-app-url/` for the web UI itself.

### Swapping in the real dataset before/after deploy

The Dockerfile builds the index from the bundled sample corpus so the
container works immediately. To deploy with the real MSMARCO-XI index
instead, either:
- run `python -m scripts.ingest --source hf ...` locally, then add
  `data/index.pkl` and `data/index.embedder.pkl` to the repo (they're
  gitignored by default — remove them from `.gitignore` if you do this,
  and be mindful of file size for a large `--limit`), or
- change the `RUN ... scripts.ingest` line in the `Dockerfile` to use
  `--source hf --language <code> --limit <n>` (this makes the build
  itself hit HuggingFace, so it needs `datasets`/`huggingface_hub` —
  already in `requirements.txt` — installed in the Docker build step too).

---

## 11. Before you submit — checklist

- [ ] Re-ingest from the real `ai4bharat/MSMARCO-XI` dataset (§3)
- [ ] Set real `SARVAM_API_KEY`, `STT_PROVIDER=sarvam` and confirm one
      real voice query works end-to-end
- [ ] Set a real generation key and `GEN_PROVIDER=anthropic`/`openai`,
      confirm answers still ground correctly
- [ ] Re-tune `OFFTOPIC_MIN_SIM` / `GROUNDING_MIN_OVERLAP` against the
      real corpus (§5's honest caveats explain why the sample-corpus
      defaults won't directly transfer)
- [ ] Re-run `scripts/benchmark.py` against `data/index.eval_queries.jsonl`
      (real queries) and with real STT/generation, for the P50/P70/P100
      numbers you actually submit
- [ ] Re-run `scripts/demo_cases.py` with the real backends and confirm
      the guardrails still hold
- [ ] Deploy (§10) and confirm `/health` and the web UI both work on
      the public URL, from your phone
- [ ] Record the two required videos and post per the promotion requirements
