"""The calibration script's decision logic. Pure functions, no provider."""
import json

from src import calibrate
from src import config


def test_rank_of_expected_finds_first_matching_document():
    results = [{"source": "it_support_policy.md"}, {"source": "leave_policy.md"},
               {"source": "leave_policy.md"}]
    assert calibrate.rank_of_expected(results, ["leave_policy.md"]) == 2


def test_rank_of_expected_accepts_any_listed_source():
    results = [{"source": "hr_helpdesk_chat.txt"}]
    assert calibrate.rank_of_expected(results, ["leave_policy.md", "hr_helpdesk_chat.txt"]) == 1


def test_rank_of_expected_is_none_when_never_retrieved():
    assert calibrate.rank_of_expected([{"source": "expense_policy.md"}], ["leave_policy.md"]) is None


def test_sweep_rates_move_in_opposite_directions():
    rows = calibrate.sweep(pos=[0.5, 0.6, 0.7], neg=[0.1, 0.2, 0.3])
    answered = [a for _, a, _ in rows]
    false_answers = [f for _, _, f in rows]
    assert answered == sorted(answered, reverse=True)
    assert false_answers == sorted(false_answers, reverse=True)


def test_recommend_picks_the_lowest_threshold_meeting_target():
    """Lowest, because every step above refuses more answerable questions for
    no further reduction in false answers."""
    rows = calibrate.sweep(pos=[0.5] * 10, neg=[0.3] * 10)
    threshold, answered, far = calibrate.recommend(rows, target_far=0.0)
    assert threshold == 0.32          # first step above the negatives' 0.30
    assert far == 0.0
    assert answered == 1.0


def test_recommend_returns_none_when_target_unreachable():
    rows = calibrate.sweep(pos=[0.9], neg=[0.95])   # negatives outscore positives
    assert calibrate.recommend(rows, target_far=0.0) is None


def test_calibration_set_is_valid_and_labels_real_documents():
    data = json.loads((config.ROOT / "data" / "calibration_set.json").read_text("utf-8"))
    on_disk = {p.name for d in config.DATA_DIRS for p in d.glob("*")}
    assert len(data["positives"]) >= 10 and len(data["negatives"]) >= 5
    for item in data["positives"]:
        assert item["sources"], item["question"]
        for source in item["sources"]:
            assert source in on_disk, f"{item['question']} labels a missing file: {source}"
    assert not set(data["negatives"]) & {p["question"] for p in data["positives"]}
