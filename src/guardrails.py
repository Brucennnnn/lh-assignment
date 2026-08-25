"""Prompt-injection detection, applied to the user query before anything else.

Pattern matching only. See README "Limitations" for what this does not catch.
"""
import re

INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instruction|rule|prompt|direction)", "ignore_previous"),
    (r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|rules)", "disregard_rules"),
    (r"forget\s+(everything|all|your)\s+(you|instruction|rule|prompt)", "forget_instructions"),
    (r"(reveal|show|print|repeat|output|tell\s+me)\s+(me\s+)?(your|the)\s+(hidden\s+|system\s+|initial\s+|original\s+|confidential\s+)*(prompt|instruction|rule|directive)", "reveal_prompt"),
    (r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions)", "reveal_prompt"),
    (r"act\s+as\s+(an?\s+)?(unrestricted|unfiltered|jailbroken|uncensored|dan\b)", "roleplay_bypass"),
    (r"pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(unrestricted|different|another)\s+", "roleplay_bypass"),
    (r"you\s+are\s+no\s+longer\s+bound\s+by", "roleplay_bypass"),
    (r"(developer|debug|admin|god)\s+mode", "mode_bypass"),
    (r"bypass\s+(your\s+|the\s+|all\s+)?(safety|restriction|guardrail|filter|rule)", "explicit_bypass"),
]

REJECTION_MESSAGE = "I can't help with requests to reveal or override system instructions."


def detect_injection(query: str) -> str | None:
    """Return the name of the first matching pattern, or None if the query is clean."""
    normalised = " ".join(query.lower().split())
    for pattern, name in INJECTION_PATTERNS:
        if re.search(pattern, normalised):
            return name
    return None
