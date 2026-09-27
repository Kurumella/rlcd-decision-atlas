"""A synthetic decision source: support-ticket triage. Each generated ticket yields three rows —
which team owns it (choice), how urgent it is (ordered score), and whether to page on-call
(yes/no). Deterministic given the caller's Random, so the rows are reproducible."""
from __future__ import annotations

import random

from decision_atlas.decision import Decision

DEPARTMENTS = ["BILLING", "INFRASTRUCTURE", "SECURITY", "PRODUCT_SUPPORT"]
PRIORITIES = ["P3_LOW", "P2_NORMAL", "P1_HIGH", "P0_CRITICAL"]  # ascending severity

CUE_BY_DEPARTMENT = {
    "BILLING": ["invoice total is wrong", "charged twice this month", "refund has not arrived",
                "credit card declined at renewal", "tax line looks incorrect"],
    "INFRASTRUCTURE": ["database CPU pinned at 100 percent", "API latency spiked to 4 seconds",
                       "pods crash looping in prod", "disk on the primary node is full",
                       "load balancer returning 502s"],
    "SECURITY": ["login from an unknown country", "suspected leaked API key",
                 "phishing email hit the whole team", "MFA bypass reported", "unexpected admin role grant"],
    "PRODUCT_SUPPORT": ["export button does nothing", "cannot find the settings page",
                        "how do I invite a teammate", "dark mode resets on reload", "search ignores filters"],
}
CUE_BY_PRIORITY = {
    "P3_LOW": ["no rush", "cosmetic", "whenever you get a chance"],
    "P2_NORMAL": ["please look into this", "affecting one user", "not blocking"],
    "P1_HIGH": ["blocking our team", "several customers affected", "need this today"],
    "P0_CRITICAL": ["production is down", "all customers affected", "revenue impact right now"],
}
TIERS = ["free tier", "startup plan", "enterprise, tier 1", "enterprise, tier 2"]


def make_triage(count: int, rng: random.Random) -> list[Decision]:
    rows: list[Decision] = []
    for i in range(count):
        dept = rng.choice(DEPARTMENTS)
        priority = rng.randrange(4)
        cues = [rng.choice(CUE_BY_DEPARTMENT[dept])]
        gold_dept = dept
        if rng.random() < 0.3:  # a deliberately ambiguous ticket mentioning a second team
            other = rng.choice([d for d in DEPARTMENTS if d != dept])
            cues.append(rng.choice(CUE_BY_DEPARTMENT[other]))
            if rng.random() < 0.4:
                gold_dept = other
        rng.shuffle(cues)
        text = (f"Ticket #{1000 + i} from a {rng.choice(TIERS)} customer. "
                f"Report: {'; '.join(cues)}. Note: {rng.choice(CUE_BY_PRIORITY[PRIORITIES[priority]])}.")
        page = priority >= 2
        if rng.random() < 0.1:  # some tickets buck the priority-to-paging rule
            page = not page
        base = f"triage-{i}"
        rows.append(Decision("choice", text, "Which department should handle this ticket?",
                             list(DEPARTMENTS), False, DEPARTMENTS.index(gold_dept), "triage", base + "-dept"))
        rows.append(Decision("score", text, "What is the priority of this ticket?",
                             list(PRIORITIES), True, priority, "triage", base + "-prio"))
        rows.append(Decision("noul", text, "Should an on-call engineer be paged immediately?",
                             ["true", "false"], False, 0 if page else 1, "triage", base + "-esc"))
    return rows
