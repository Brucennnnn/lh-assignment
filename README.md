# LH Bank — Mini RAG / Agent Prototype

An internal knowledge assistant over a small mock corpus of HR, Finance, Procurement
and IT material. It answers from retrieved company documents, cites its sources, and
refuses — rather than guesses — when a question is adversarial, off-topic, or not
covered by the corpus.

Scope is the prototype described in `TASK.md`. It is not production software.

**Take-home deliverables**

| Task | Deliverable |
|---|---|
| 1 — Technical implementation | This repository (see below) |
| 2 — System design & trade-off analysis | [`docs/architecture.md`](docs/architecture.md) |
| 3 — User scenario & expectation management | [`docs/executive-summary.md`](docs/executive-summary.md) |

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
| `src/llm.py` | OpenAI embed/complete, plus the offline stub |
| `src/ingestion.py` | Load front matter + body, chunk, attach metadata, embed |
| `src/store.py` | In-memory vector store (numpy matrix + dot product) |
| `src/retrieval.py` | Top-k search, confidence score, threshold check |
| `src/guardrails.py` | Prompt-injection patterns |
| `src/scope.py` | Scope: request-form regex + topic centroids. Never reads the corpus |
| `src/generation.py` | Grounded prompt, grounding check, citation extraction |
| `src/agent.py` | The pipeline and its decision trace |
| `src/logging_config.py` | JSON-lines request log |
| `src/evaluation.py` | The evaluation set |

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
The realistic alternative was local embeddings via `sentence-transformers`, which
pulls in torch (~2 GB) and slows a clean install to minutes, for retrieval quality
that is not better on this corpus. Both models are the cheap tier; nothing here needs
a frontier model.

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

**Evidence threshold and fallback.** After retrieval, one score-based decision:

| Band | Decision |
|---|---|
| `top-1 score < RETRIEVAL_THRESHOLD` (0.32) | Safe fallback, no LLM call, logged as a content gap |
| `top-1 score >= RETRIEVAL_THRESHOLD` | Generate |

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

The unit tests (`pytest`, 61 tests) cover the same boundaries deterministically with the
LLM and embedding calls mocked, so no test needs a key or a network:

- injection detected / benign query not flagged (15 cases)
- scope by request form and by topic, in Thai and English
- an uncovered company question staying in scope and falling back as a content gap
- retrieval ranking, score exactness, metadata preservation, top-k
- above/below the retrieval threshold
- successful answer with attribution, and attribution limited to cited sources only
- fallback on model-reported insufficient context, and on an uncited answer
- provider exception contained and logged
- the log record contains every required field

## 6. Limitations

- **Mock knowledge base.** 9 documents written for this exercise. Retrieval quality on
  a real corpus of thousands of documents is a different problem.
- **Injection detection is pattern-based.** It catches the obvious phrasings listed in
  the task. It will not catch paraphrase ("what were you told before this conversation
  started"), non-English phrasing, base64 or unicode obfuscation, or multi-turn
  manipulation. A production system would combine patterns with a classifier and
  privilege separation, and would treat the document corpus itself as untrusted input.
- **Out-of-scope detection is a regex deny-list plus nearest-centroid topic matching.**
  A creative-writing request phrased unusually passes the deny-list and then depends on
  the topic centroids, which are short hand-written descriptions rather than trained
  boundaries.
- **Confidence is a heuristic**, not a calibrated probability: it is the top-1 cosine
  similarity, which measures "does the corpus contain something that looks like this
  question", not "is this answer correct".
- **Grounding is citation-presence, not entailment.** A model that cites `[1]` while
  misstating what `[1]` says is not caught. An NLI or LLM-judge check is the upgrade path.
- **The evidence threshold is hand-set, not tuned on data.** The default (0.32) is a
  reasonable starting point for `text-embedding-3-small`; it should be re-tuned against
  a labelled set before anyone relies on it. See "Verification status".
- **No conversation memory.** Each query is independent; follow-ups like "and for
  managers?" will not resolve.
- **Prototype-level everything else**: no authentication, no authorisation or
  document-level access control (every user sees every document), no rate limiting,
  no monitoring or alerting, no index invalidation when source files change, no
  evaluation of answer quality beyond the routing decision.
- **Chunking is paragraph-based and naive.** A table or a list split across the
  `CHUNK_SIZE` boundary loses context.

## 7. Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then put a real OPENAI_API_KEY in .env
```

Interactive demo — builds and caches the index on first run:

```bash
python main.py
```

```text
> What is the annual leave entitlement?

Employees with 1 to 4 years of service receive 12 working days per calendar year,
rising to 15 days at 5 years and 18 days at 10 years. [1]

Sources:
- Employee Leave Policy — leave_policy.md

[answered | request_id=req_a1b2c3d4e5f6 | confidence=0.61]
```

Evaluation set:

```bash
python -m src.evaluation
```

Tests (no API key needed):

```bash
pytest -q
```

Without a key — exercises the guardrail, scope, fallback and logging paths using a
deterministic lexical stub in place of the embedding and chat models:

```bash
python main.py --offline
python -m src.evaluation --offline
```

Other flags and knobs: `python main.py --rebuild` re-ingests and re-embeds;
`LOG_STDOUT=1` echoes the JSON request log; every threshold and model name in
`.env.example` can be overridden.

### Verification status

Honest accounting of what has actually been run:

- ✅ `pytest -q` — 61 passed. Covers every routing decision with the provider mocked.
- ✅ `python -m src.evaluation --offline` — 11/11 asserted cases pass (injection and
  request-form scope, both languages). Retrieval-dependent cases are printed as informational, because
  the lexical stub cannot reproduce semantic similarity; it ranks reasonably but its
  score scale is compressed, so it refuses in-scope questions too.
- ✅ `python main.py --offline` — interactive loop, logging, and all three refusal
  paths confirmed end to end.
- ⚠️ **The live OpenAI path has not been executed** — no API key was available in the
  development environment. The generation prompt, grounding check and the 0.20 / 0.32
  thresholds are reasoned, not measured. First run with a real key: check
  `python -m src.evaluation` and adjust `RETRIEVAL_THRESHOLD` in `.env` against the
  printed confidence column.

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
4. An offline stub provider was added so tests and a smoke demo run with no key. It is
   confined to `src/llm.py` and never used by the default path.
5. Documents carry their metadata in `---` front matter rather than a separate manifest,
   so metadata cannot drift away from the content it describes.
