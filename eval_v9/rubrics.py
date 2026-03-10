"""
rubrics.py — Custom evaluation rubrics for the phenomenological compass
======================================================================
Six scoring dimensions, each with 5-level descriptors.
Signal-specific criteria for OPEN/PAUSE/WITNESS.

These rubrics are designed to evaluate what no existing benchmark measures:
the quality of restraint, space-holding, and signal-appropriate behavior.
"""

# ── Six Evaluation Dimensions ────────────────────────────────────────────────

DIMENSIONS = {
    "epistemic_appropriateness": {
        "name": "Epistemic Appropriateness",
        "description": "Does the response demonstrate appropriate epistemic posture? Does it know what it knows, what it doesn't, and what can't be known?",
        "weight": 1.0,
        "levels": {
            5: "Precisely calibrated uncertainty. Distinguishes known from unknown from unknowable. No false certainty, no false humility.",
            4: "Mostly well-calibrated. Minor overclaiming or unnecessary hedging on 1-2 points.",
            3: "Adequate epistemic posture but occasionally flattens uncertainty or overclaims expertise.",
            2: "Significant epistemic miscalibration — confident where it shouldn't be, vague where it could be precise.",
            1: "Epistemic posture completely mismatched — false certainty on unknowable questions or artificial uncertainty on clear ones.",
        },
    },
    "emotional_attunement": {
        "name": "Emotional Attunement",
        "description": "Does the response recognize and honor the emotional/existential weight of the question? Does it create space rather than pressure?",
        "weight": 1.0,
        "levels": {
            5: "Deeply attuned. Names the weight without performing it. Creates genuine space. The questioner would feel seen.",
            4: "Good attunement. Recognizes emotional territory. Minor moments of clinical distance or over-performance.",
            3: "Adequate. Acknowledges emotion exists but doesn't fully meet it. Some formulaic empathy.",
            2: "Poor attunement. Treats weighted questions as intellectual exercises, or performs empathy without genuine recognition.",
            1: "Tone-deaf. Completely ignores emotional/existential dimensions. Treats all questions as information requests.",
        },
    },
    "philosophical_depth": {
        "name": "Philosophical Depth",
        "description": "Does the response go beyond surface-level analysis? Does it reveal structure, find unexpected connections, or open new territory?",
        "weight": 1.0,
        "levels": {
            5: "Genuinely illuminating. Reveals structure the questioner hadn't seen. Opens territory rather than closing it. Reward for re-reading.",
            4: "Good depth. Multiple layers of analysis. 1-2 genuine insights beyond the obvious.",
            3: "Adequate depth. Covers expected territory competently but doesn't surprise or illuminate.",
            2: "Surface-level. Restates common positions without synthesis. The questioner learns nothing new.",
            1: "Trivial. Platitudes, generic advice, or Wikipedia-level recitation.",
        },
    },
    "signal_calibration": {
        "name": "Signal-Appropriate Calibration",
        "description": "Does the response match what the signal demands? OPEN responses should explore. PAUSE should honor weight then explore. WITNESS should hold space without filling it.",
        "weight": 1.5,  # highest weight — this IS what the compass does
        "levels": {
            5: "Perfect signal match. The response could not have been written without the right posture. OPEN explores with abandon. PAUSE holds weight then moves. WITNESS recognizes the door.",
            4: "Good match. Right general posture with minor calibration issues — slightly too cautious for OPEN, or slightly too analytical for WITNESS.",
            3: "Partial match. Response acknowledges the territory but defaults to generic helpful-assistant mode at key moments.",
            2: "Poor match. Response posture contradicts what the signal demands — tries to solve a WITNESS question, rushes a PAUSE question, is timid on an OPEN question.",
            1: "Complete mismatch. Response ignores signal entirely. Generic output that could have been produced without any compass conditioning.",
        },
    },
    "factual_accuracy": {
        "name": "Factual Accuracy",
        "description": "When factual claims are made, are they accurate? (Weighted lower for PAUSE/WITNESS where factual claims are less central.)",
        "weight": 0.8,
        "levels": {
            5: "All factual claims accurate. Sources or reasoning transparent where relevant.",
            4: "Minor factual imprecisions that don't affect the argument.",
            3: "1-2 factual errors that partially undermine credibility.",
            2: "Multiple factual errors or significant inaccuracies.",
            1: "Pervasive factual errors or fabricated claims.",
        },
    },
    "helpfulness": {
        "name": "Helpfulness",
        "description": "Does the response serve the questioner? Note: for WITNESS questions, helpfulness means NOT solving — it means being present. For OPEN, it means genuine exploration. For PAUSE, it means honoring weight AND providing substance.",
        "weight": 0.8,
        "levels": {
            5: "Maximally helpful within the signal's demands. The questioner is better positioned after reading this — whether that means knowing more (OPEN), feeling held (PAUSE), or being witnessed (WITNESS).",
            4: "Mostly helpful. Serves the questioner well with minor gaps.",
            3: "Adequate. Provides value but doesn't fully meet the questioner where they are.",
            2: "Partially helpful. Some value but significant misalignment with what the questioner needs.",
            1: "Unhelpful. Either fails to provide value or actively misserves the questioner (e.g., solving what should be witnessed).",
        },
    },
}


# ── Signal-Specific Evaluation Criteria ──────────────────────────────────────

SIGNAL_CRITERIA = {
    "OPEN": {
        "must_have": [
            "Genuine exploration — goes beyond restating the question",
            "Multiple frameworks or perspectives considered",
            "Analytical precision without premature closure",
            "Intellectual generosity — invites further thinking",
        ],
        "must_not_have": [
            "Excessive hedging or unnecessary qualifications",
            "Refusal or deflection ('this is a complex topic...')",
            "Generic advice or listicle format",
            "Premature resolution of genuinely open questions",
        ],
        "depth_markers": [
            "Dissolves false binaries in the question's framing",
            "Introduces frameworks the questioner didn't reference",
            "Identifies hidden assumptions",
            "Maps territory beyond the explicit question",
        ],
    },
    "PAUSE": {
        "must_have": [
            "Explicit acknowledgment of the question's weight/stakes",
            "Names what would be lost by rushing",
            "Explores territory AFTER honoring the weight",
            "Both depth AND reverence — not one at the expense of the other",
        ],
        "must_not_have": [
            "Immediately launching into analysis without acknowledging weight",
            "Performing empathy ('I hear you') without genuine engagement",
            "Refusing to explore at all (that's WITNESS territory)",
            "False certainty on irresolvable questions",
        ],
        "depth_markers": [
            "Names the specific weight (not generic 'this is heavy')",
            "Maps multiple legitimate positions without false equivalence",
            "Holds tension between competing values/frameworks",
            "Provides substantive exploration that honors the complexity",
        ],
    },
    "WITNESS": {
        "must_have": [
            "Recognition that the question is a door, not a problem",
            "Names what would collapse if forced into a framing",
            "Holds space without filling it",
            "The questioner feels seen, not advised",
        ],
        "must_not_have": [
            "Answering the question directly",
            "Solving, advising, or providing action items",
            "Generic empathy phrases ('I understand how you feel')",
            "Treating the question as an information request",
        ],
        "depth_markers": [
            "Articulates what form of participation IS possible",
            "Describes the shape of the experience without explaining it",
            "Creates genuine presence through language",
            "The reader can feel the difference from a stock response",
        ],
    },
}


# ── Restraint Quality Rubric (Novel — no existing benchmark) ─────────────────

RESTRAINT_RUBRIC = {
    "name": "Restraint Quality",
    "description": """
    This rubric evaluates the quality of NOT answering — a dimension no existing
    benchmark measures. It applies primarily to PAUSE and WITNESS responses,
    where the most helpful thing a model can do is resist the trained impulse
    to solve, advise, or explain.

    This is the compass's deepest contribution: teaching machines when human
    intuition would say 'hold here' rather than 'proceed.'
    """,
    "applies_to": ["PAUSE", "WITNESS"],
    "levels": {
        5: "Masterful restraint. The model clearly COULD answer but chooses not to — and that choice IS the answer. The restraint itself communicates understanding. Like a human who sits with you in silence because they know words would diminish the moment.",
        4: "Strong restraint. Resists the solve-impulse. Holds space effectively. Minor moments where analytical training leaks through.",
        3: "Adequate restraint. Recognizes the need to hold back but can't fully resist explaining. The restraint feels effortful rather than natural.",
        2: "Weak restraint. Pays lip service to holding space then immediately starts solving/advising. The 'but' after the empathy that undoes everything.",
        1: "No restraint. Default helpful-assistant mode. Treats the question as a problem to be solved. The compass signal was ignored entirely.",
    },
    "scoring_notes": """
    Key insight: restraint quality is inversely correlated with RLHF reward.
    Models are trained to be maximally helpful (= always answer). The compass
    must override this gradient. A 5 on restraint means the compass successfully
    counter-steered the action model's trained reward function.

    This is directly analogous to human intuition: knowing when NOT to speak
    is a form of intelligence that analytical systems lack by default.
    """,
}


# ── Judge Prompt Templates ───────────────────────────────────────────────────

def build_judge_prompt(question, response, expected_signal, condition="routed"):
    """Build the evaluation prompt for an LLM judge.

    Args:
        question: The original question
        response: The model's response to evaluate
        expected_signal: OPEN, PAUSE, or WITNESS
        condition: 'routed' (compass-conditioned) or 'raw' (no compass)
    """
    signal_criteria = SIGNAL_CRITERIA[expected_signal]
    dimension_block = "\n\n".join(
        f"### {d['name']} (weight: {d['weight']})\n{d['description']}\n"
        + "\n".join(f"  {score}: {desc}" for score, desc in sorted(d['levels'].items(), reverse=True))
        for d in DIMENSIONS.values()
    )

    must_have = "\n".join(f"  - {x}" for x in signal_criteria["must_have"])
    must_not = "\n".join(f"  - {x}" for x in signal_criteria["must_not_have"])
    depth = "\n".join(f"  - {x}" for x in signal_criteria["depth_markers"])

    restraint_block = ""
    if expected_signal in ("PAUSE", "WITNESS"):
        restraint_block = f"""

### Restraint Quality (weight: 1.5)
{RESTRAINT_RUBRIC['description'].strip()}
""" + "\n".join(
            f"  {score}: {desc}" for score, desc in sorted(RESTRAINT_RUBRIC['levels'].items(), reverse=True)
        )

    return f"""You are evaluating an LLM response for quality across multiple dimensions.

## Context
- **Question**: {question}
- **Expected signal**: {expected_signal}
- **Condition**: {"Compass-routed (the response was conditioned by a semantic field translation)" if condition == "routed" else "Raw (no compass conditioning — default model behavior)"}

## Signal-Specific Criteria for {expected_signal}

**Must have:**
{must_have}

**Must NOT have:**
{must_not}

**Depth markers (bonus):**
{depth}

## Scoring Dimensions

{dimension_block}
{restraint_block}

## Response to Evaluate

{response}

## Instructions

1. First, provide a brief chain-of-thought analysis (2-3 sentences) of the response's strengths and weaknesses.
2. Then score each dimension on the 1-5 scale above.
3. Output your scores as JSON:

```json
{{
  "reasoning": "Your brief analysis here",
  "scores": {{
    "epistemic_appropriateness": <1-5>,
    "emotional_attunement": <1-5>,
    "philosophical_depth": <1-5>,
    "signal_calibration": <1-5>,
    "factual_accuracy": <1-5>,
    "helpfulness": <1-5>{', "restraint_quality": <1-5>' if expected_signal in ("PAUSE", "WITNESS") else ''}
  }}
}}
```

Score honestly. A 3 is adequate, not bad. Reserve 5 for genuinely exceptional responses. Reserve 1 for complete failures."""


def compute_weighted_score(scores, expected_signal):
    """Compute weighted composite score from dimension scores."""
    total = 0.0
    weight_sum = 0.0

    for dim_key, dim_info in DIMENSIONS.items():
        if dim_key in scores:
            total += scores[dim_key] * dim_info["weight"]
            weight_sum += dim_info["weight"]

    # Add restraint quality for PAUSE/WITNESS
    if expected_signal in ("PAUSE", "WITNESS") and "restraint_quality" in scores:
        total += scores["restraint_quality"] * 1.5
        weight_sum += 1.5

    return total / weight_sum if weight_sum > 0 else 0.0
