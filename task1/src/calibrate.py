"""Pick RETRIEVAL_THRESHOLD and TOP_K from data instead of guessing.

    python -m src.calibrate                    # real provider
    python -m src.calibrate --offline          # stub, only checks the script runs
    python -m src.calibrate --target-far 0.02  # allow a 2% false-answer rate

Reads `data/calibration_set.json`: questions the corpus provably answers
(positives, each labelled with the documents that answer it) and questions it
provably does not (negatives). Both lists are the SMEs' to own.

The threshold is not a mathematical optimum, it is a business decision about
which mistake you would rather make. Set it too low and the system answers from
irrelevant documents, which the user cannot detect. Set it too high and it
refuses questions it could have answered, which the user can. This script makes
that trade explicit: it reports what every candidate threshold costs in both
directions and recommends the lowest one that keeps false answers under target.
"""
import argparse
import json

from . import llm
from . import config
from . import ingestion
from .retrieval import retrieve

MAX_K = 10
BUCKETS = 20


def load_set() -> tuple[list[dict], list[str]]:
    data = json.loads((config.ROOT / "data" / "calibration_set.json").read_text("utf-8"))
    return data["positives"], data["negatives"]


def measure(store, questions: list[str]) -> list[list[dict]]:
    """Retrieve MAX_K for each question. One embedding call for the whole batch."""
    vectors = llm.embed(questions)
    return [retrieve(store, v, k=MAX_K) for v in vectors]


def top_score(results: list[dict]) -> float:
    return results[0]["score"] if results else 0.0


def rank_of_expected(results: list[dict], sources: list[str]) -> int | None:
    """1-based rank of the first chunk from a document that answers the question."""
    for i, r in enumerate(results, start=1):
        if r["source"] in sources:
            return i
    return None


def histogram(scores: list[float], label: str, lo: float, hi: float) -> None:
    width = (hi - lo) / BUCKETS or 1.0
    counts = [0] * BUCKETS
    for s in scores:
        counts[min(int((s - lo) / width), BUCKETS - 1)] += 1
    peak = max(counts) or 1
    print(f"\n{label}  (n={len(scores)})")
    for i, c in enumerate(counts):
        bar = "#" * round(c / peak * 44)
        print(f"  {lo + i * width:5.2f} | {bar}{' ' if c else ''}{c or ''}")


def sweep(pos: list[float], neg: list[float]) -> list[tuple[float, float, float]]:
    """For each candidate threshold: (threshold, answered rate, false-answer rate)."""
    rows = []
    for step in range(0, 96, 2):
        t = step / 100
        rows.append((t,
                     sum(s >= t for s in pos) / len(pos),
                     sum(s >= t for s in neg) / len(neg)))
    return rows


def recommend(rows, target_far: float) -> tuple[float, float, float] | None:
    """Lowest threshold whose false-answer rate meets target - lowest, because
    every step above it refuses more answerable questions for no further gain."""
    for t, answered, far in rows:
        if far <= target_far:
            return t, answered, far
    return None


def run(offline: bool = False, target_far: float = 0.01) -> int:
    if offline:
        llm.use_stub()
        config.apply_stub_thresholds()
        store = ingestion.build_index(persist=False)
    else:
        store = ingestion.load_index()

    positives, negatives = load_set()
    print(f"corpus: {len(store.chunks)} chunks · "
          f"{len(positives)} answerable questions · {len(negatives)} unanswerable\n")

    pos_results = measure(store, [p["question"] for p in positives])
    neg_results = measure(store, negatives)
    pos_scores = [top_score(r) for r in pos_results]
    neg_scores = [top_score(r) for r in neg_results]

    # --- 1. the two distributions ---
    lo, hi = min(pos_scores + neg_scores), max(pos_scores + neg_scores)
    histogram(neg_scores, "UNANSWERABLE questions - top-1 score", lo, hi)
    histogram(pos_scores, "ANSWERABLE questions - top-1 score", lo, hi)

    overlap_lo, overlap_hi = min(pos_scores), max(neg_scores)
    if overlap_lo > overlap_hi:
        print(f"\nThe two groups do not overlap: any threshold between "
              f"{overlap_hi:.2f} and {overlap_lo:.2f} separates them perfectly.")
    else:
        share = sum(overlap_lo <= s <= overlap_hi for s in pos_scores + neg_scores)
        print(f"\nOverlap region {overlap_lo:.2f} - {overlap_hi:.2f}, containing "
              f"{share} of {len(pos_scores) + len(neg_scores)} questions. "
              f"No threshold separates these; the trade below is unavoidable.")

    # --- 2. what each threshold costs ---
    rows = sweep(pos_scores, neg_scores)
    print("\nthreshold   answerable answered   unanswerable answered (bad)")
    print("-" * 62)
    pick = recommend(rows, target_far)
    for t, answered, far in rows:
        if answered == 0 and far == 0:
            break
        mark = "  <-- recommended" if pick and abs(t - pick[0]) < 1e-9 else ""
        print(f"   {t:.2f}       {answered:6.1%}              {far:6.1%}{mark}")

    print(f"\ncurrent RETRIEVAL_THRESHOLD = {config.RETRIEVAL_THRESHOLD}")
    if pick:
        t, answered, far = pick
        print(f"recommended                 = {t:.2f}   "
              f"(answers {answered:.0%} of answerable questions, "
              f"{far:.0%} false answers, target {target_far:.0%})")
        if answered < 0.80:
            print("\n  WARNING: meeting the false-answer target costs more than 20% of\n"
                  "  answerable questions. The threshold is not the problem - the two\n"
                  "  groups overlap too much. Fix retrieval first: chunk size, overlap,\n"
                  "  a multilingual embedding model, or hybrid BM25 + vector search.")
    else:
        print(f"recommended                 = none reachable at target "
              f"{target_far:.0%}; unanswerable questions score too high. "
              f"Fix retrieval before tuning the threshold.")

    # --- 3. how many candidates to retrieve ---
    ranks = [rank_of_expected(r, p["sources"]) for r, p in zip(pos_results, positives)]
    print(f"\nRecall@k - is a document that answers the question in the top k?")
    print("-" * 62)
    previous = 0.0
    for k in range(1, MAX_K + 1):
        recall = sum(r is not None and r <= k for r in ranks) / len(ranks)
        knee = "  <-- gains flatten here" if previous and recall - previous < 0.02 else ""
        print(f"   k={k:<3}      {recall:6.1%}{knee}")
        if knee:
            break
        previous = recall
    print(f"\ncurrent TOP_K = {config.TOP_K}")

    missed = [p["question"] for p, r in zip(positives, ranks) if r is None]
    if missed:
        print(f"\nNever retrieved at k={MAX_K} - these are retrieval failures, not "
              f"threshold problems:")
        for q in missed:
            print(f"  - {q}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                        help="use the deterministic stub (checks the script runs, "
                             "cannot produce usable numbers)")
    parser.add_argument("--target-far", type=float, default=0.01,
                        help="acceptable false-answer rate on unanswerable questions")
    args = parser.parse_args()
    raise SystemExit(run(args.offline, args.target_far))
