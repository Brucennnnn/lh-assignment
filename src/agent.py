"""The pipeline. Every stage can end the request; every ending is logged.

query -> injection guard -> scope precheck -> retrieval -> confidence
      -> generation -> grounding check -> answer + sources
"""
import time
import uuid

from . import generation, guardrails, ingestion, logging_config, retrieval, scope

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

        reason = scope.precheck(query)
        if reason:
            record["scope"] = "out_of_scope"
            record["scope_reason"] = reason
            record["status"] = OUT_OF_SCOPE
            return {"answer": scope.REJECTION_MESSAGE}

        results = retrieval.retrieve(self.store, query)
        record["retrieved_sources"] = [r["source"] for r in results]
        record["retrieval_scores"] = [r["score"] for r in results]
        record["retrieval_confidence"] = retrieval.confidence(results)

        reason = scope.classify_by_score(record["retrieval_confidence"])
        if reason:
            record["scope"] = "out_of_scope"
            record["scope_reason"] = reason
            record["status"] = OUT_OF_SCOPE
            return {"answer": scope.REJECTION_MESSAGE}
        record["scope"] = "in_scope"

        if not retrieval.is_confident(results):
            record["fallback"] = True
            record["fallback_reason"] = "retrieval_below_threshold"
            record["status"] = FALLBACK
            return {"answer": generation.FALLBACK_MESSAGE}

        answer, failure = generation.generate(query, results)
        if failure:
            record["fallback"] = True
            record["fallback_reason"] = failure
            record["generation_status"] = "rejected"
            record["status"] = FALLBACK
            return {"answer": generation.FALLBACK_MESSAGE}

        record["generation_status"] = "ok"
        record["status"] = ANSWERED
        sources = generation.cited_sources(answer, results)
        record["cited_sources"] = [s["source"] for s in sources]
        return {"answer": answer, "sources": sources}
