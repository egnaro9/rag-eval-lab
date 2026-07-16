"""A tiny, unambiguous demo corpus + eval set.

Each document is short enough to become a single chunk (``<doc_id>#0``), which
keeps the gold labels obvious. The corpus is deliberately factual so retrieval
either clearly works or clearly does not.
"""
from __future__ import annotations

from typing import Dict, List

SAMPLE_DOCS: Dict[str, str] = {
    "mercury": (
        "Mercury is the smallest planet in the Solar System and the closest to the Sun. "
        "It has almost no atmosphere and a heavily cratered surface."
    ),
    "venus": (
        "Venus is the second planet from the Sun and the hottest planet in the Solar System. "
        "Its thick carbon dioxide atmosphere traps heat in a runaway greenhouse effect."
    ),
    "earth": (
        "Earth is the third planet from the Sun and the only known planet to support life. "
        "About seventy one percent of its surface is covered by water."
    ),
    "mars": (
        "Mars is the fourth planet from the Sun and is called the Red Planet because of iron oxide. "
        "It hosts Olympus Mons, the tallest volcano in the Solar System."
    ),
    "jupiter": (
        "Jupiter is the largest planet in the Solar System and is a gas giant of hydrogen and helium. "
        "Its Great Red Spot is a storm larger than Earth."
    ),
    "saturn": (
        "Saturn is the sixth planet from the Sun and is famous for its ring system of ice and rock. "
        "It is the second largest planet in the Solar System."
    ),
}

# Gold chunk ids are ``<doc_id>#0`` because each doc fits in one chunk.
EVAL_SET: List[dict] = [
    {"q": "Which planet is the hottest in the Solar System?", "gold_ids": ["venus#0"]},
    {"q": "What is the tallest volcano in the Solar System?", "gold_ids": ["mars#0"]},
    {"q": "Which planet is famous for its ring system?", "gold_ids": ["saturn#0"]},
    {"q": "What is the largest planet?", "gold_ids": ["jupiter#0"]},
    {"q": "Which planet is closest to the Sun?", "gold_ids": ["mercury#0"]},
]

# A deliberately-wrong answer used to prove the eval harness catches
# hallucinations end-to-end. Its content words (neptune, volcanic, geysers)
# do not appear in the correctly-retrieved Venus context, so faithfulness
# drops below threshold and the case is flagged.
PLANTED = {
    "q": "Which planet is the hottest in the Solar System?",
    "gold_ids": ["venus#0"],
    "hallucinated_answer": "Neptune is the hottest planet because of its volcanic geysers.",
    "note": "PLANTED hallucination — answer is not supported by the retrieved context.",
}
