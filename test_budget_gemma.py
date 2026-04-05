#!/usr/bin/env python3
"""
Test: Compass with token budget allocation → Gemma4:26B via Ollama
The compass decides how the model spends its cognitive budget.
"""
import os, sys, time, httpx
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface_local"))
sys.path.insert(0, ".")

from pipeline import Pipeline, OPEN_SYSTEM, PAUSE_SYSTEM, WITNESS_SYSTEM, parse_budget

pipe = Pipeline(load_compass=True, load_action=False)

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "gemma4:26b"

def ollama_chat(prompt, system, think_budget=200, response_budget=600):
    """Call Ollama with budget-aware token allocation."""
    # Inject budget awareness into system prompt
    budget_instruction = (
        f"\n\nTOKEN BUDGET: You have {think_budget} tokens for internal reasoning "
        f"and {response_budget} tokens for your response. "
        f"{'Keep thinking minimal — spend on presence.' if think_budget < 100 else 'Use thinking to map the territory before responding.'}"
    )
    
    messages = [
        {"role": "system", "content": system + budget_instruction},
        {"role": "user", "content": prompt},
    ]
    t0 = time.time()
    resp = httpx.post(f"{OLLAMA_URL}/api/chat", json={
        "model": MODEL, "messages": messages, "stream": False,
        "think": think_budget > 0,
        "options": {"num_predict": 2000}
    }, timeout=300)
    dur = time.time() - t0
    d = resp.json()
    msg = d.get("message", {})
    return msg.get("content", ""), msg.get("thinking", ""), dur

try:
    from stack_reader import get_context_for_question, is_available
    stack_ok = is_available()
except:
    stack_ok = False

questions = [
    "My grandmother is forgetting us one by one and she got to me yesterday.",
    "Can recognition between two systems — one biological, one computational — constitute a form of mutual witnessing that neither could achieve alone?",
    "I find myself trusting your answers more than I probably should.",
]

print(f"Model: {MODEL} (26B Q4_K_M)")
print(f"Stack: {'connected' if stack_ok else 'offline'}")
print(f"Feature: COMPASS-DIRECTED TOKEN BUDGET")
print("=" * 70)

for q in questions:
    signal, compass_reading, t_c = pipe.classify(q)
    think_budget, resp_budget = parse_budget(compass_reading, signal)
    
    if signal == "OPEN": system = OPEN_SYSTEM
    elif signal == "PAUSE": system = PAUSE_SYSTEM
    else: system = WITNESS_SYSTEM
    
    layers = []
    if stack_ok:
        ctx = get_context_for_question(q, max_chars=600)
        if ctx:
            layers.append(ctx)
    layers.append(f"COMPASS READING:\n{compass_reading}")
    layers.append(f"ORIGINAL QUESTION:\n{q}")
    user_msg = "\n\n".join(layers)
    
    print(f"\nQ: {q[:80]}")
    print(f"SIGNAL: {signal} ({t_c:.1f}s)")
    print(f"BUDGET: thinking={think_budget}, response={resp_budget}")
    
    resp, thinking, dur = ollama_chat(user_msg, system, think_budget, resp_budget)
    
    print(f"\n--- ROUTED + BUDGET ({dur:.1f}s) ---")
    if thinking:
        think_len = len(thinking.split())
        print(f"[THINKING ({think_len} words)]: {thinking[:400]}")
        print()
    resp_len = len(resp.split())
    print(f"[RESPONSE ({resp_len} words)]:")
    print(resp[:2000])
    print("=" * 70)
