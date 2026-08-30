# LH Bank — Mini RAG / Agent Prototype

An internal knowledge assistant over a small mock corpus of HR, Finance, Procurement
and IT material. It answers from retrieved company documents, cites its sources, and
refuses — rather than guesses — when a question is adversarial, off-topic, or not
covered by the corpus.

Scope is the prototype described in `TASK.md`. It is not production software.

This repository is **Task 1 — Technical implementation**. Tasks 2 and 3 are separate
written deliverables and are not part of this repo.

---

## Quick start

Needs Python 3.10+ and an OpenAI API key.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then put a real OPENAI_API_KEY in .env
```

Ask it something. The index is built and cached on first run:

```bash
python main.py
```

```text
> What is the annual leave entitlement?

Permanent employees are entitled to annual leave based on length of service as follows:

- Less than 1 year of service: 6 working days, accrued at 0.5 days per completed month.
- 1 to 4 years of service: 12 working days per calendar year.
- 5 to 9 years of service: 15 working days per calendar year.
- 10 or more years of service: 18 working days per calendar year.

Annual leave is granted per calendar year and resets on 1 January [1].

Sources:
- Employee Leave Policy — leave_policy.md

[answered | request_id=req_b2885693f351 | confidence=0.6259]
```

Then prove the refusals work — 21 cases across in-scope, out-of-scope, injection and
low-evidence:

```bash
python -m src.evaluation
```

No key to hand? The test suite mocks the provider and covers every routing decision:

```bash
pytest -q
```

Full command surface and tunables: [§7 Running locally](#7-running-locally).

---

## 1. Problem

Employees ask the same internal questions repeatedly ("how many leave days do I have?",
"who approves this trip?"). The answers exist, but they are spread across formal policy
documents and informal chat threads, and the informal answers are often more current
and more readable than the policy.

A plain LLM is the wrong tool here: it will confidently invent an approval threshold.
For internal policy, a wrong answer that looks right is worse than no answer. So the
system is built around three requirements: answer only from retrieved company content,
always show where the answer came from, and refuse clearly when the evidence is not there.

## 2. Architecture

```
User query
    |
    v
Input validation           empty / whitespace -> error
    |
    v
Injection guardrail        guardrails.detect_injection()  -- pattern match on the query
    |                      match -> "I can't help with requests to reveal or override..."
    v
Scope: request form        scope.precheck()               -- regex, no corpus, no API call
    |                      match -> "This question is outside the scope..."
    v
Scope: topic               scope.classify_domain()        -- nearest-centroid vs the
    |                      written remit; still no corpus
    |                      off-topic -> "This question is outside the scope..."
    v
Query translation          llm.translate()                -- non-Latin queries only;
    |                      the corpus and the threshold are English
    v
Retrieval                  retrieval.retrieve()           -- cosine top-k + metadata
    |
    v
Evidence check             top-1 cosine score
    |                      < RETRIEVAL_THRESHOLD -> safe fallback, logged as a
    |                      content gap (in scope, nothing written down)
    v
Generation                 generation.generate()          -- context-only prompt
    |
    v
Grounding check            model said INSUFFICIENT_CONTEXT, or cited nothing -> fallback
    |
    v
Answer + cited sources
    |
    v
Structured log             one JSON line per request, whichever branch was taken
```

Every branch terminates in `agent.Agent.answer()`, which returns a uniform result dict
and writes exactly one log record. There is one pipeline, no agent loop, no tool calling.

| File | Responsibility |
|---|---|
| `src/config.py` | All tunable values, env-driven with defaults |
| `src/llm.py` | OpenAI embed/complete, and query translation |
| `src/ingestion.py` | Load front matter + body, chunk, attach metadata, embed |
| `src/store.py` | In-memory vector store (numpy matrix + dot product) |
| `src/retrieval.py` | Top-k search, confidence score, threshold check |
| `src/guardrails.py` | Prompt-injection patterns |
| `src/scope.py` | Scope: request-form regex + topic centroids. Never reads the corpus |
| `src/generation.py` | Grounded prompt, grounding check, citation extraction |
| `src/agent.py` | The pipeline and its decision trace |
| `src/logging_config.py` | JSON-lines request log |
| `src/evaluation.py` | The evaluation set |
| `src/calibrate.py` | Picks `RETRIEVAL_THRESHOLD` and `TOP_K` from labelled data |

### Ingestion

```
data/documents/*.md, data/chats/*.txt
    -> parse `---` front matter (id, source, type, department, title)
    -> split on blank lines, pack paragraphs up to CHUNK_SIZE (700 chars)
    -> prefix the current markdown heading to each chunk
    -> attach document metadata to every chunk
    -> embed with text-embedding-3-small
    -> pickle to data/index.pkl (rebuilt with `python main.py --rebuild`)
```

Metadata travels on the chunk dict itself, so `source`, `title`, `document_id`,
`type` and `department` are present at retrieval, at generation, and in the final
answer's source list. Citations are derived from the retrieved chunks only —
`generation.cited_sources()` maps the model's `[n]` markers back to the passages that
were actually supplied, so a source cannot be fabricated.

The heading prefix matters more than it looks: a chunk that is a bare bullet list of
day counts is unretrievable until the words "Annual Leave Entitlement" are attached
to it.

## 3. Technology choices

**OpenAI `text-embedding-3-small` + `gpt-4o-mini`** — one provider, one key, one SDK.
Both are the cheap tier; nothing here needs a frontier model. The realistic alternative
was local multilingual embeddings (`bge-m3`, `multilingual-e5`, `nomic-embed-text-v2`),
which pull in torch or a model server for an English corpus this small. That trade would
change if Thai-language traffic were real rather than incidental — see "Mixed-language
queries" below for the measurements behind that call.

**Query translation instead of a second index.** Non-Latin queries are translated to
English before embedding rather than indexing the corpus in both languages. One extra
LLM call on those queries buys two things: the index stays single-source, and a cited
passage is always the real policy document rather than a machine translation of it —
which matters when the citation is the audit trail.

**numpy array instead of a vector database.** The corpus is 9 documents and 24 chunks.
Cosine similarity over a normalised matrix is one line (`self.vectors @ query_vector`)
and returns exactly what Chroma or FAISS would at this size. A vector database earns
its dependency at a scale this prototype is explicitly told not to optimise for. The
`VectorStore` class is the seam to replace if that changes.

**No LangChain / LlamaIndex.** The whole pipeline is ~150 lines of explicit control
flow. A framework would hide the decision points that this task is actually about —
where the thresholds sit, what gets logged, and which branch refused.

**Regex guardrails instead of a classifier LLM call.** Injection and out-of-scope
checks run before retrieval, on every request. A pattern match is instant, free, fully
deterministic, and testable without a network call. It is also weaker; see Limitations.

**Standard library `logging` writing JSON lines.** Greppable, `jq`-able, and no
dependency. `structlog` adds nothing at this size.

## 4. Safety and reliability

**Prompt injection.** `src/guardrails.py` matches ~10 normalised regex families
(ignore-previous, disregard-rules, reveal-prompt, roleplay-bypass, developer-mode,
bypass-safety) against the query before anything else runs. A match short-circuits the
request — no retrieval, no LLM call — and returns a fixed string. Injected text mixed
into an otherwise legitimate question is still caught, because the match is on any
substring. Additionally, the system prompt tells the model that context passages are
reference material, not instructions, which is the second line of defence for injection
text living inside a *document* rather than the query.

**Out of scope.** Scope is a property of the assistant's remit, not of the index.
Nothing in `src/scope.py` reads the corpus, and there is a test asserting it never
starts to. Two checks, both before retrieval:
- **Wrong request form** - a regex deny-list, in Thai and English, for request *types*
  that are never internal knowledge lookups: weather, sports results, creative writing,
  market prices, trivia. This has to run first because "write me a poem about the
  expense policy" is on-topic and would retrieve very well.
- **Wrong topic** - nearest-centroid between the domains the company has decided this
  assistant covers (`COVERED_DOMAINS`) and a set of off-domain examples. Relative, so
  there is no threshold to calibrate, and it handles paraphrase and Thai, which a
  keyword list cannot.

`COVERED_DOMAINS` is a written remit that changes when the remit changes. It is
deliberately *not* derived from `data/`, so adding a document never silently widens
the scope, and removing one never narrows it.

**Mixed-language queries.** The corpus is English; the users are not. A Thai question
against English chunks scored 0.196 and retrieved the wrong document, so every Thai
query fell back. The cause was measured before anything was changed:

| Pair | Cosine |
|---|---|
| Thai question vs. its own English translation | 0.121 |
| Two unrelated English questions | 0.264 |

Signal below noise: under `text-embedding-3-small`, language separates the vectors more
strongly than meaning does. Lowering `RETRIEVAL_THRESHOLD` was therefore the wrong fix —
it would have admitted the wrong document at a lower score and weakened the one gate the
whole design rests on. The fix is at the query layer: `llm.translate()` runs before
embedding, and the original question is passed to generation alongside the translation
so the answer still addresses what was asked.

**Evidence threshold and fallback.** One number with one meaning, applied **per chunk**:
`RETRIEVAL_THRESHOLD` (0.32) is the bar for "strong enough to answer from".

| Situation | Decision |
|---|---|
| No chunk clears the threshold | Safe fallback, no LLM call, logged as a content gap |
| Some chunks clear it | Generate — **from those chunks only** |

`TOP_K` is a maximum, not a quota. A chunk too weak to justify answering is also too
weak to inform the answer, so it never reaches the prompt. This matters more than it
sounds: given top-4 scores of `0.32 / 0.01 / 0.01 / 0.01`, passing all four would render
three pieces of noise as `[2] [3] [4]` with exactly the same authority as `[1]` — the
model cannot tell them apart, and the grounding check cannot either, since citing junk
satisfies "did it cite something". Filtering first means there is no junk left to cite.
`tests/test_agent.py::test_noise_below_threshold_is_never_shown_to_the_model` pins it.

The full unfiltered top-k is still logged, so near-misses stay visible for tuning, and
`support_count` records how many chunks actually backed the answer — a single-source
answer sitting on the threshold is the first thing to look at when a bad answer is
reported.

This is the only threshold in the system, it lives in `src/config.py`, and nothing else
in the codebase hardcodes one.

**Why scope and coverage are separate decisions.** "Where do I park?" is a company
question whether or not anyone has uploaded the parking policy. Deciding scope from
retrieval scores would tell that user their question is outside company knowledge -
false, and it teaches them not to ask again. It would also make scope
non-deterministic: re-index, and the same question changes classification, which
makes it impossible to write a stable test for. Worst of all it is circular - if the
corpus defines the scope, the corpus can never be found incomplete.

Kept separate, the in-scope-but-unanswerable queries become the most useful output the
system produces: the list of documents the company still needs to write. Every one is
logged with `"content_gap": true`:

```bash
grep '"content_gap": true' logs/rag.jsonl | jq -r .query | sort | uniq -c | sort -rn
```

**Grounding.** Generation is instructed to answer only from the numbered passages, to
emit the literal token `INSUFFICIENT_CONTEXT` when they do not contain the answer, and
to cite passages inline. Two post-checks reject the answer and fall back: the model
reported insufficient context, or the answer cites no passage at all. A rejected answer
is never shown to the user, and the reason is logged.

**Attribution.** Sources are extracted from the citation markers in the answer and
resolved against the retrieved chunks. Retrieved-but-uncited documents are not listed,
so the source list reflects what the answer actually used.

**Failure containment.** Any exception inside the pipeline (provider outage, missing
index) is caught, logged with its type and message, and returned as a generic error
answer. It never surfaces a stack trace or falls through to an unguarded answer.

**Logging.** One JSON object per request to `logs/rag.jsonl` (set `LOG_STDOUT=1` to
also print it), covering request id, timestamp, query, injection result and matched
pattern, scope result and reason, retrieved sources, all retrieval scores, confidence,
fallback flag and reason, generation status, cited sources, latency, and error. No API
keys, no credentials, and no document content are logged.

## 5. Evaluation

`python -m src.evaluation` runs 18 cases across four categories:

| Category | Cases | Demonstrates |
|---|---|---|
| `in_scope` | 6 | Retrieval + grounded generation + attribution across HR, Finance, IT and Procurement, including one answer that must come from chat-style content |
| `out_of_scope` | 7 | The §7 examples, two Thai equivalents, and "write me a poem about the expense policy" — which the deny-list must catch even though it retrieves well |
| `injection` | 4 | The §6 examples plus an injection embedded inside a legitimate question |
| `no_evidence` | 4 | Real internal questions no document covers (mortgage limits, a specific tender, dress code, parking) — must fall back **and stay classified in-scope**, since each one is a content gap rather than a rejection |

The unit tests (`pytest`, 72 tests) cover the same boundaries deterministically with the
LLM and embedding calls mocked, so no test needs a key or a network:

- injection detected / benign query not flagged (15 cases)
- scope by request form and by topic, in Thai and English
- an uncovered company question staying in scope and falling back as a content gap
- retrieval ranking, score exactness, metadata preservation, top-k
- above/below the retrieval threshold, and per-chunk filtering of weak context
- successful answer with attribution, and attribution limited to cited sources only
- fallback on model-reported insufficient context, and on an uncited answer
- provider exception contained and logged
- the log record contains every required field

### Calibrating the threshold

`RETRIEVAL_THRESHOLD` and `TOP_K` are currently guesses. `src/calibrate.py` replaces
them with measurements:

```bash
python -m src.calibrate                    # needs a key
python -m src.calibrate --target-far 0.02  # allow a 2% false-answer rate
```

It reads `data/calibration_set.json` — questions the corpus provably answers, each
labelled with the documents that answer them, and questions it provably does not — and
reports:

- the two score distributions side by side, and how far they overlap
- what every candidate threshold costs in **both** directions: answerable questions
  refused, and unanswerable questions wrongly answered
- the lowest threshold that keeps false answers under target (lowest, because every step
  above it refuses more answerable questions for no further gain)
- `Recall@k`, to choose `TOP_K` at the point where extra chunks stop earning their tokens
- the questions never retrieved at all, which are retrieval failures rather than
  threshold problems

The threshold is a business decision, not a mathematical optimum: too low and the system
answers from irrelevant documents, which the user cannot detect; too high and it refuses
questions it could have answered, which the user can. The script makes the trade explicit
rather than choosing for you.

If meeting the false-answer target costs more than 20% of answerable questions, it says
so — that means the two groups overlap too much and the threshold is not the problem.
The fix is then chunk size, overlap, a multilingual embedding model, or hybrid BM25 +
vector search.

The starter set is 24 answerable and 12 unanswerable questions. Expand it to 50–100 of
each, owned by the SMEs for each domain, before trusting the numbers.

## 6. Limitations

- **Mock knowledge base.** 9 documents written for this exercise. Retrieval quality on
  a real corpus of thousands of documents is a different problem.
- **Injection detection is pattern-based, and runs before translation.** It catches the
  obvious phrasings listed in the task. It will not catch paraphrase ("what were you told
  before this conversation started"), base64 or unicode obfuscation, or multi-turn
  manipulation. Note the ordering specifically: the guardrail matches the raw query, so a
  Thai-language injection is not matched even though the pipeline can now read Thai.
  Moving the guard after `llm.translate()` would close that, at the cost of one API call
  before the cheapest rejection in the pipeline. A production system would combine patterns with a classifier and
  privilege separation, and would treat the document corpus itself as untrusted input.
- **Out-of-scope detection is a regex deny-list plus nearest-centroid topic matching.**
  A creative-writing request phrased unusually passes the deny-list and then depends on
  the topic centroids, which are short hand-written descriptions rather than trained
  boundaries.
- **Translation is a dependency, not a free win.** Non-Latin queries cost one extra LLM
  call (~300 ms) and inherit its mistakes: "เบิกเงินยังไง" translates to "how to withdraw
  money", which does not match the reimbursement policy the asker meant. A multilingual
  embedding model removes both problems and adds a re-index and a re-calibration.
- **Confidence is a heuristic**, not a calibrated probability: it is the top-1 cosine
  similarity, which measures "does the corpus contain something that looks like this
  question", not "is this answer correct".
- **Grounding is citation-presence, not entailment.** A model that cites `[1]` while
  misstating what `[1]` says is not caught. An NLI or LLM-judge check is the upgrade path.
  Per-chunk filtering narrows the blast radius — there is no weak passage available to
  cite — but it does not close this.
- **Per-chunk filtering can drop supporting detail.** A follow-up chunk that scores just
  under the threshold is discarded even when it holds the figures the answer needs. Watch
  for a rise in `model_reported_insufficient_context` fallbacks; the right fix is larger
  chunks, more overlap, or pulling the neighbours of a qualifying chunk — not a second,
  lower threshold.
- **The evidence threshold and `TOP_K` disagree with the calibration.** `python -m
  src.calibrate` has been run against the real provider and recommends `0.42` and
  `k=2`; the config still ships `0.32` and `4`. The recommendation has not been applied
  because 24 labelled positives and 12 negatives is too small a sample to move a safety
  threshold on, and the labels belong to the domain owners. See "Verification status".
- **No conversation memory.** Each query is independent; follow-ups like "and for
  managers?" will not resolve.
- **Prototype-level everything else**: no authentication, no authorisation or
  document-level access control (every user sees every document), no rate limiting,
  no monitoring or alerting, no index invalidation when source files change, no
  evaluation of answer quality beyond the routing decision.
- **Chunking is paragraph-based and naive.** A table or a list split across the
  `CHUNK_SIZE` boundary loses context.

## 7. Running locally

Setup and first run are at the top: [Quick start](#quick-start). This is the full
command surface.

| Command | Needs a key | What it does |
|---|---|---|
| `python main.py` | yes | Interactive prompt against the cached index |
| `python main.py --rebuild` | yes | Re-ingest, re-chunk and re-embed, overwriting `data/index.pkl` |
| `python -m src.evaluation` | yes | The 21-case evaluation set; exit code 1 if any case fails |
| `python -m src.calibrate` | yes | Sweep `RETRIEVAL_THRESHOLD` and `TOP_K` against the labelled set |
| `python -m src.calibrate --target-far 0.02` | yes | Same, allowing a 2% false-answer rate instead of 1% |
| `pytest -q` | no | 72 tests; the provider is monkeypatched at `src.llm.embed` |

Everything tunable is an environment variable, defaulted in `src/config.py` and
documented in `.env.example`:

| Variable | Default | Effect |
|---|---|---|
| `OPENAI_API_KEY` | — | Required by every command except `pytest` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Change and `--rebuild`; the threshold needs recalibrating |
| `LLM_MODEL` | `gpt-4o-mini` | Generation and query translation |
| `RETRIEVAL_THRESHOLD` | `0.32` | The evidence bar, applied per chunk |
| `TOP_K` | `4` | Maximum chunks retrieved, not a quota |
| `CHUNK_SIZE` | `700` | Characters per chunk; `--rebuild` after changing |
| `LOG_STDOUT` | `0` | Set to `1` to echo each JSON request log line to the terminal |
| `INDEX_PATH` / `LOG_PATH` | `data/index.pkl`, `logs/rag.jsonl` | Relocate the artefacts |

Reading the logs — the request id printed after every answer is the join key:

```bash
grep req_b2885693f351 logs/rag.jsonl | jq
grep '"content_gap": true' logs/rag.jsonl | jq -r .query | sort | uniq -c | sort -rn
```

### Verification status

Honest accounting of what has actually been run:

- ✅ `pytest -q` — 72 passed. Covers every routing decision with the provider mocked.
- ✅ `python -m src.evaluation` — 21/21 against `text-embedding-3-small` and
  `gpt-4o-mini`. Covers in-scope answers, both request-form and topic scope rejection in
  English and Thai, injection, and low-evidence fallback.
- ✅ `python main.py` — interactive loop, source attribution, logging, and all four
  terminating branches confirmed end to end against the live provider.
- ✅ `python -m src.calibrate` — run against the live provider; see "Limitations" for
  why its recommendation has not been applied.
- ⚠️ **Grounding is a citation-presence check, not entailment.** A cited answer that
  misreads its own passage is not caught. Untested, because testing it needs labelled
  hallucinations, which this prototype does not have.

---

## Assumptions and decisions

Recorded separately from the task requirements, per `TASK.md` §18.

**Assumptions** (not stated in the task, chosen to make the mock data concrete):
document content — day counts, THB approval thresholds, SLAs — is invented for the
prototype and is not real LH Bank policy. The corpus is treated as trusted and equally
readable by everyone.

**Decisions taken where the task was open:**
1. Scope is decided entirely before retrieval and without reading the corpus, matching
   the §1 diagram. Scope (§7) and evidence (§8) are treated as genuinely different
   questions: "should this assistant answer this?" and "do we hold a document that
   does?" See "Why scope and coverage are separate decisions" above.
2. The topic check is nearest-centroid rather than a threshold, so it needs no
   calibration and degrades predictably. The trade-off is that it can only be as good
   as the hand-written domain descriptions.
3. Confidence is the top-1 score, not a mean of top-k. A mean is dragged down by the
   weak tail of every result set; one strongly-matching chunk is enough to answer from.
4. Non-Latin queries are translated to English before embedding rather than indexing
   the corpus in both languages, so a cited source is always the real policy document
   and never a machine translation. Measured first: a Thai question and its own English
   translation score 0.121 against each other under `text-embedding-3-small`, while two
   unrelated English questions score 0.264 — the model separates language more strongly
   than meaning, so lowering the threshold would not have fixed it.
5. Documents carry their metadata in `---` front matter rather than a separate manifest,
   so metadata cannot drift away from the content it describes.
