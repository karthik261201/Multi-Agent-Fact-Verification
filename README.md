# Verification Agent

This is Agent 3 in the pipeline:

```
Agent 1 (Claim Analysis) -> Agent 2 (Evidence Retrieval) -> Agent 3 (Verification, this folder)
```

It takes a claim + a list of evidence items (the format your Evidence Agent
already outputs, as in your screenshot: `Evidence N / Relevance / Title / URL
/ snippet`) and returns a verdict: **SUPPORTED**, **REFUTED**, or
**INSUFFICIENT**, with an explanation grounded in the evidence.

## Files

```
verification_agent/
├── agent.py              # the verification agent itself
├── requirements.txt      # one dependency: anthropic
├── sample_evidence.json  # example input (structured JSON)
├── sample_evidence.txt   # example input (raw text, same shape as your log)
└── README.md
```

## 1. Set up the folder

Copy this whole `verification_agent/` folder into your team's project repo,
next to (not inside) the folders your teammates are using for the claim
agent and evidence agent, e.g.:

```
project/
├── claim_agent/
├── evidence_agent/
└── verification_agent/   <- this one
```

## 2. Install dependencies

From inside `verification_agent/`:

```bash
pip install -r requirements.txt
```

## 3. Get a free API key (no credit card)

This project uses **Groq**, which has a genuine free tier — no credit card,
no trial expiry, just an email signup.

1. Go to https://console.groq.com and sign up with your email.
2. In the console, go to **API Keys > Create API Key**, name it, and copy it.
3. Set it as an environment variable:

```bash
# macOS / Linux
export GROQ_API_KEY="your-key-here"

# Windows (PowerShell)
$env:GROQ_API_KEY="your-key-here"
```

The free tier is rate-limited (roughly 30 requests/min and a daily token
cap), but that's more than enough for a student project — you just can't
hammer it with hundreds of calls per minute.

## 4. Run it

With the structured JSON sample:

```bash
python agent.py --input sample_evidence.json
```

With a raw text dump (same shape your Evidence Agent already prints to
console/log, as in your screenshot) — you pass the claim separately since raw
text files don't carry it:

```bash
python agent.py --input sample_evidence.txt --claim "OpenAI is losing money on its ChatGPT Pro subscription plan."
```

Expected output:

```
Claim: OpenAI is losing money on its ChatGPT Pro subscription plan.
Loaded 3 evidence item(s). Verifying...

============================================================
VERDICT: SUPPORTED
Confidence: 0.9
Supporting evidence: [3, 4]
Contradicting evidence: []

Explanation:
Evidence items 3 and 4 both directly quote CEO Sam Altman stating OpenAI
is losing money on the $200/month ChatGPT Pro plan due to heavier-than-
modeled usage. Evidence 5 supports the broader context of company-wide
losses but is less directly on-point.
============================================================
```

## 5. Wiring it into the full pipeline

When your teammates' Evidence Agent is ready, instead of reading from a
file you can just import and call the two core functions directly from your
own orchestrator code:

```python
from verification_agent.agent import call_verification_model, Evidence

evidence_list = [
    Evidence(id=i, title=e["title"], url=e["url"],
             relevance=e["relevance"], snippet=e["snippet"])
    for i, e in enumerate(evidence_agent_output, start=1)
]

result = call_verification_model(claim_text, evidence_list)
print(result.verdict, result.explanation)
```

`result` is a `VerificationResult` object with `.verdict`, `.explanation`,
`.supporting_evidence_ids`, `.contradicting_evidence_ids`, and `.confidence`
— easy to pass along to whatever orchestrator/UI stitches the three agents
together (matching the diagram: verdict → explanation, or back to the
orchestrator to search again if INSUFFICIENT).

## Notes

- Swap `MODEL = "openai/gpt-oss-120b"` in `agent.py` for a different model
  string if you want to try another one available on Groq's free tier
  (e.g. `llama-3.3-70b-versatile`, `qwen/qwen3-32b`).
- The agent is instructed to base its verdict **only** on the evidence given
  to it (not outside knowledge), which is what makes the verdict
  evidence-grounded and explainable rather than just "the model's opinion."
- Other genuinely free, no-card options if you ever want to compare: Google
  AI Studio (Gemini 2.5 Flash, very generous daily limit) and OpenRouter
  (several free open models). Swapping providers just means changing the
  `import` and the `client.chat.completions.create(...)` call to match that
  provider's SDK — the rest of the script (parsing, prompt, CLI) stays the
  same.
