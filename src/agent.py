"""The pipeline. Every stage can end the request; every ending is logged.

query -> injection guard -> scope (request form, then topic) -> retrieval
      -> evidence check -> generation -> grounding check -> answer + sources

Scope is settled before retrieval and without consulting the corpus, so
"we don't answer that" and "nobody has written that down yet" stay separate
outcomes. The second is logged as a content gap; the list of those is the
knowledge base's to-do list.
"""
import time
import uuid

from . import generation, guardrails, ingestion, llm, logging_config, retrieval, scope

# Statuses returned to the caller.
ANSWERED = "answered"
BLOCKED_INJECTION = "blocked_injection"
OUT_OF_SCOPE = "out_of_scope"
FALLBACK = "fallback"
ERROR = "error"


class Agent:
    def __init__(self, store=None):
        self.store = store or ingestion.load_index()

    def answer(self, query: str) -> dict:
        started = time.perf_counter()
        record = {
            "request_id": f"req_{uuid.uuid4().hex[:12]}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "query": query,
            "injection_detected": False,
            "scope": None,
            "content_gap": False,
            "support_count": 0,
            "retrieved_sources": [],
            "retrieval_scores": [],
            "retrieval_confidence": None,
            "fallback": False,
            "fallback_reason": None,
            "generation_status": None,
            "status": None,
            "error": None,
        }
        result = {"answer": None, "sources": []}

        try:
            result.update(self._run(query, record))
        except Exception as exc:  # provider outage, bad index, etc.
            record["status"] = ERROR
            record["error"] = f"{type(exc).__name__}: {exc}"
            result["answer"] = (
                "Something went wrong while answering. The error has been logged."
            )

        record["latency_ms"] = round((time.perf_counter() - started) * 1000)
        logging_config.log_request(record)
        return {**result, "status": record["status"], "request_id": record["request_id"],
                "confidence": record["retrieval_confidence"], "trace": record}

    def _run(self, query: str, record: dict) -> dict:
        if not query or not query.strip():
            record["status"] = ERROR
            record["error"] = "empty_query"
            return {"answer": "Please enter a question."}

        injection = guardrails.detect_injection(query)
        if injection:
            record["injection_detected"] = True
            record["injection_pattern"] = injection
            record["status"] = BLOCKED_INJECTION
            return {"answer": guardrails.REJECTION_MESSAGE}

        # Scope, both halves, before the corpus is touched.
        reason = scope.precheck(query)
        if reason:
            record["scope"] = "out_of_scope"
            record["scope_reason"] = reason
            record["status"] = OUT_OF_SCOPE
            return {"answer": scope.REJECTION_MESSAGE}

        # One embedding call, used for the topic check and then for retrieval.
        query_vector = llm.embed([query])[0]

        reason = scope.classify_domain(query_vector)
        if reason:
            record["scope"] = "out_of_scope"
            record["scope_reason"] = reason
            record["status"] = OUT_OF_SCOPE
            return {"answer": scope.REJECTION_MESSAGE}
        record["scope"] = "in_scope"

        results = retrieval.retrieve(self.store, query_vector)
        record["retrieved_sources"] = [r["source"] for r in results]
        record["retrieval_scores"] = [r["score"] for r in results]
        record["retrieval_confidence"] = retrieval.confidence(results)

        # Only chunks that clear the threshold reach the model. Everything below
        # it is noise the model cannot distinguish from evidence, and would cite.
        evidence = retrieval.qualifying(results)
        record["support_count"] = len(evidence)

        if not evidence:
            # In scope, but the corpus does not cover it. Not the user's mistake
            # and not a rejection - a gap in the knowledge base, recorded as one.
            record["fallback"] = True
            record["fallback_reason"] = "no_evidence_in_corpus"
            record["content_gap"] = True
            record["status"] = FALLBACK
            return {"answer": generation.FALLBACK_MESSAGE}

        answer, failure = generation.generate(query, evidence)
        if failure:
            record["fallback"] = True
            record["fallback_reason"] = failure
            record["generation_status"] = "rejected"
            record["status"] = FALLBACK
            return {"answer": generation.FALLBACK_MESSAGE}

        record["generation_status"] = "ok"
        record["status"] = ANSWERED
        sources = generation.cited_sources(answer, evidence)
        record["cited_sources"] = [s["source"] for s in sources]
        return {"answer": answer, "sources": sources}
