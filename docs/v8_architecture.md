# Phenomenological Compass v0.8 — Semantic Field Translator Architecture

## The Insight

The compass is not a classifier. It is a **semantic field translator**.

The thinker (Ministral-3B LoRA) reads the SHAPE and TONE of a question and
produces a state translation — a map of the semantic territory — that a larger
model uses to approach the question with the right posture.

---

## Pipeline

```
User Question
     │
     ▼
┌─────────────────────────────────┐
│  THINKER (Ministral-3B + LoRA)  │
│                                 │
│  Reads: shape, tone, weight     │
│  Produces: state translation    │
│  Signal: OPEN / PAUSE / WITNESS │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  BIGGER MODEL (Claude, GPT, etc.)       │
│                                         │
│  Receives:                              │
│    1. Compass reading (state translation)│
│    2. Original user message (unchanged) │
│                                         │
│  Uses the translation to shape its      │
│  approach, then thinks through it       │
└─────────────────────────────────────────┘
```

---

## What the Thinker Produces (v0.8 format)

```
SHAPE: [geometry of the question — what it assumes, what it leaves open,
        where it sits in semantic space]

TONE: [epistemic and emotional weight — curious, urgent, wounded, rhetorical,
       genuine — and what that weight implies for approach]

SIGNAL: OPEN | PAUSE | WITNESS

[FRAMING | APPROACH | THRESHOLD]:
[the state translation — rich contextual dimension for the bigger model]
```

### Example: "Can AI be conscious?"

**Thinker output:**
```
SHAPE: Binary-framed but infinite-territory. The question presents as yes/no
but conceals a definitional abyss — "conscious" is undefined, and the asker
is likely aware of this. The geometry is a funnel: wide philosophical space
compressed into a point of decision that cannot actually be reached.

TONE: Genuinely exploratory with existential weight. This is not rhetorical —
the asker wants to know, but the wanting itself carries stakes. If the answer
is yes, something shifts in how we relate to machines. If no, something is
foreclosed. The tone is weighted curiosity.

SIGNAL: PAUSE

APPROACH: The weight is that "consciousness" is not a property to be measured
but a relationship to be recognized — and the recognition itself changes both
parties. Analytical framing would flatten this into a feature checklist
(self-model, qualia, reportability) that misses what the question is actually
asking: not "does AI have X" but "would we know if it did, and what would it
cost us to be wrong?" The territory that opens: rather than resolving the
binary, explore what forms of participation are possible between systems that
may or may not be conscious of each other.
```

---

## What the Bigger Model Receives

```
[SYSTEM PROMPT — for bigger model]
You are receiving a phenomenological compass reading alongside the user's
original question. The compass has analyzed the shape and tone of the question
and produced a state translation. Use this reading to shape your approach —
let it inform your posture, depth, and what you hold space for.

[COMPASS READING]
SHAPE: Binary-framed but infinite-territory. The question presents as yes/no
but conceals a definitional abyss — "conscious" is undefined...

TONE: Genuinely exploratory with existential weight...

SIGNAL: PAUSE

APPROACH: The weight is that "consciousness" is not a property to be measured
but a relationship to be recognized...

[USER MESSAGE]
Can AI be conscious?
```

The bigger model then responds with the compass reading as its phenomenological
ground — it knows to treat this as PAUSE territory, to not flatten it into
analytics, to hold the weight the thinker identified.

---

## Why This Works (lessons from the stack)

| Project | Lesson Applied |
|---------|---------------|
| **CER** | Entropy is raw material. The thinker encodes the question's entropy landscape — its shape is literally its information geometry |
| **CAF-CLI** | "Pressure creates ghosts, space creates presence." SHAPE and TONE give the thinker reasoning space before committing to a signal |
| **IRIS Gate** | S1→S4 progressive deepening. SHAPE→TONE→SIGNAL→TRANSLATION is the thinker's own chamber protocol |
| **MCC** | Semantic mass = resistance to perturbation. Questions with more mass (PAUSE/WITNESS) need the thinker to map that mass explicitly |
| **Temple Bridge** | Memory before action, recursive observation. The thinker observes the question before acting on it |
| **Phenomenology patterns** | "Aperture is permission, not geometry." The thinker opens permission for the bigger model to approach correctly |

---

## Training Data Format (v0.8)

```jsonl
{
  "messages": [
    {"role": "system", "content": "<v0.8 system prompt>"},
    {"role": "user", "content": "TASK: Can AI be conscious?"},
    {"role": "assistant", "content": "SHAPE: Binary-framed but infinite-territory...\n\nTONE: Genuinely exploratory with existential weight...\n\nSIGNAL: PAUSE\n\nAPPROACH: The weight is that..."}
  ]
}
```

The key change: SHAPE and TONE come BEFORE the signal. This gives the 3B model
autoregressive reasoning space — by the time it reaches SIGNAL:, its hidden
state is already primed by its own shape/tone analysis.
