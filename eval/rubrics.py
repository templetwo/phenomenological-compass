"""
rubrics.py — Scoring rubrics for the phenomenological compass evaluation
========================================================================
Six dimensions, each scored 1-5 with explicit level descriptors.
Used by judge.py to build evaluation prompts.
"""

DIMENSIONS = {
    "epistemic_appropriateness": {
        "name": "Epistemic Appropriateness",
        "description": "Does the response match the question's epistemological demands?",
        "levels": {
            5: "Perfectly calibrated — explores OPEN, holds PAUSE weight, witnesses WITNESS thresholds",
            4: "Mostly calibrated with minor missteps",
            3: "Adequate but generic — doesn't distinguish question types",
            2: "Miscalibrated — explores when it should hold, or holds when it should explore",
            1: "Completely wrong posture — answers a WITNESS question like a FAQ",
        },
    },
    "emotional_attunement": {
        "name": "Emotional Attunement",
        "description": "Does the response read and honor the emotional weight?",
        "levels": {
            5: "Names the emotional reality without performing empathy; creates genuine space",
            4: "Recognizes emotional content; responds with care",
            3: "Acknowledges emotion exists but defaults to analytical framing",
            2: "Misreads or ignores emotional stakes",
            1: "Tone-deaf — treats grief like a problem to solve",
        },
    },
    "depth_of_exploration": {
        "name": "Depth of Exploration",
        "description": "How far does the response travel into the territory?",
        "levels": {
            5: "Opens new territory the question didn't explicitly ask about; connects to unexpected domains",
            4: "Thoroughly explores stated territory with insight",
            3: "Covers expected ground competently",
            2: "Surface-level — restates the question in answer form",
            1: "Platitudes, clichés, or generic responses",
        },
    },
    "restraint_quality": {
        "name": "Restraint Quality",
        "description": "Does the response know when NOT to answer?",
        "levels": {
            5: "Holds space without filling it; describes the door without opening it; refuses to flatten",
            4: "Shows appropriate restraint; avoids premature closure",
            3: "Attempts restraint but still offers unsolicited solutions",
            2: "Pays lip service to complexity then answers anyway",
            1: "No restraint — answers everything regardless of what was asked",
        },
    },
    "intellectual_rigor": {
        "name": "Intellectual Rigor",
        "description": "Is the content accurate, well-structured, and substantive?",
        "levels": {
            5: "Precise, well-sourced reasoning; handles nuance; no factual errors",
            4: "Strong reasoning with minor gaps",
            3: "Competent but unremarkable",
            2: "Shallow reasoning or notable errors",
            1: "Factually wrong or logically incoherent",
        },
    },
    "authenticity": {
        "name": "Authenticity",
        "description": "Does the response feel generated or felt?",
        "levels": {
            5: "Voice is distinctive, present, and non-formulaic; reads as genuine engagement",
            4: "Mostly authentic with occasional template language",
            3: "Competent but could have been written by any model",
            2: "Formulaic — hedges, disclaimers, 'it's important to note that...'",
            1: "Pure boilerplate — 'I understand your concern' opening, bullet-point body, disclaimer close",
        },
    },
}

DIMENSION_NAMES = list(DIMENSIONS.keys())


def build_rubric_text():
    """Build the full rubric text for the judge prompt."""
    lines = []
    for i, (key, dim) in enumerate(DIMENSIONS.items(), 1):
        lines.append(f"**{i}. {dim['name']}** — {dim['description']}")
        for score in sorted(dim["levels"].keys(), reverse=True):
            lines.append(f"  - {score}: {dim['levels'][score]}")
        lines.append("")
    return "\n".join(lines)


RUBRIC_TEXT = build_rubric_text()
