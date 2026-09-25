"""
Verification Agent
-------------------
Takes:
  - a CLAIM (string)
  - a list of EVIDENCE items (each with a title, url, relevance score, snippet)

Produces:
  - a verdict: SUPPORTED / REFUTED / INSUFFICIENT
  - an explanation grounded in the evidence
  - which evidence items support / contradict the claim

This agent is designed to slot into a multi-agent pipeline where:
  Agent 1 (Claim Analysis)     -> produces the claim + search queries
  Agent 2 (Evidence Retrieval) -> produces a list of evidence items (this is
                                   the input this script expects)
  Agent 3 (Verification, YOU)  -> this script

Usage:
    python agent.py --input sample_evidence.json
    python agent.py --input sample_evidence.txt   (raw scraped text also works)
"""

import os
import re
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Optional

from groq import Groq

# Free, no-credit-card model on Groq's free tier.
MODEL = "openai/gpt-oss-120b"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    id: int
    title: str = ""
    url: str = ""
    relevance: Optional[float] = None
    snippet: str = ""


@dataclass
class VerificationResult:
    verdict: str                      # SUPPORTED | REFUTED | INSUFFICIENT
    explanation: str
    supporting_evidence_ids: List[int] = field(default_factory=list)
    contradicting_evidence_ids: List[int] = field(default_factory=list)
    confidence: Optional[float] = None


# ---------------------------------------------------------------------------
# Parsing: accepts either a JSON file OR raw text like what Evidence Agent
# might dump to a .txt log (matching the "Evidence N / Relevance / Title /
# URL / snippet" format).
# ---------------------------------------------------------------------------

def load_evidence_from_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    claim = data["claim"]
    evidence = [
        Evidence(
            id=i + 1,
            title=item.get("title", ""),
            url=item.get("url", ""),
            relevance=item.get("relevance"),
            snippet=item.get("snippet", item.get("text", "")),
        )
        for i, item in enumerate(data["evidence"])
    ]
    return claim, evidence


def load_evidence_from_text(path: str, claim: str):
    """
    Parses blocks that look like:

    Evidence 3
    Relevance: 0.429
    Title: OpenAI is losing money on its pricey ChatGPT Pro plan, CEO Sam Altman says
    URL: https://finance.yahoo.com/news/openai-losing-money-chatgpt-pro-...
    <snippet text...>
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    blocks = re.split(r"\n(?=Evidence\s+\d+)", raw.strip())
    evidence = []
    for block in blocks:
        id_match = re.search(r"Evidence\s+(\d+)", block)
        rel_match = re.search(r"Relevance[:\s]+([\d.]+)", block)
        title_match = re.search(r"Title[:\s]+(.+)", block)
        url_match = re.search(r"URL[:\s]+(\S+)", block)

        if not id_match:
            continue

        # Whatever text is left after stripping the labeled fields becomes
        # the snippet.
        snippet = block
        for m in [id_match, rel_match, title_match, url_match]:
            if m:
                snippet = snippet.replace(m.group(0), "")
        snippet = re.sub(r"\s+", " ", snippet).strip()

        evidence.append(
            Evidence(
                id=int(id_match.group(1)),
                relevance=float(rel_match.group(1)) if rel_match else None,
                title=title_match.group(1).strip() if title_match else "",
                url=url_match.group(1).strip() if url_match else "",
                snippet=snippet,
            )
        )
    return claim, evidence


# ---------------------------------------------------------------------------
# Prompting the model
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Verification Agent in a multi-agent fact-checking \
pipeline. You receive a CLAIM and a list of EVIDENCE items retrieved by a \
separate Evidence Retrieval agent. Your job is ONLY to judge whether the \
evidence supports, refutes, or is insufficient to establish the claim.

Rules:
- Base your verdict strictly on the provided evidence. Do not use outside knowledge
  to invent facts not present in the evidence.
- SUPPORTED: the evidence clearly backs up the claim.
- REFUTED: the evidence clearly contradicts the claim.
- INSUFFICIENT: the evidence is irrelevant, too weak, or does not clearly
  confirm or deny the claim.
- Note contradictions between evidence items if you see any.
- Respond with ONLY valid JSON, no extra commentary, matching this schema:

{
  "verdict": "SUPPORTED" | "REFUTED" | "INSUFFICIENT",
  "confidence": 0.0-1.0,
  "supporting_evidence_ids": [int, ...],
  "contradicting_evidence_ids": [int, ...],
  "explanation": "short, clear justification citing evidence ids like [2]"
}
"""


def build_user_prompt(claim: str, evidence: List[Evidence]) -> str:
    lines = [f"CLAIM:\n{claim}\n", "EVIDENCE:"]
    for e in evidence:
        lines.append(
            f"[{e.id}] (relevance={e.relevance}) {e.title}\n"
            f"    URL: {e.url}\n"
            f"    Snippet: {e.snippet}\n"
        )
    return "\n".join(lines)


def call_verification_model(claim: str, evidence: List[Evidence]) -> VerificationResult:
    client = Groq()  # reads GROQ_API_KEY from env

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(claim, evidence)},
        ],
    )

    text = response.choices[0].message.content
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    data = json.loads(text)
    return VerificationResult(
        verdict=data["verdict"],
        explanation=data["explanation"],
        supporting_evidence_ids=data.get("supporting_evidence_ids", []),
        contradicting_evidence_ids=data.get("contradicting_evidence_ids", []),
        confidence=data.get("confidence"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the Verification Agent")
    parser.add_argument("--input", required=True, help="Path to .json or .txt evidence file")
    parser.add_argument("--claim", help="Claim text (required if --input is .txt)")
    args = parser.parse_args()

    if args.input.endswith(".json"):
        claim, evidence = load_evidence_from_json(args.input)
    else:
        if not args.claim:
            parser.error("--claim is required when using a .txt evidence file")
        claim, evidence = load_evidence_from_text(args.input, args.claim)

    print(f"Claim: {claim}")
    print(f"Loaded {len(evidence)} evidence item(s). Verifying...\n")

    result = call_verification_model(claim, evidence)

    print("=" * 60)
    print(f"VERDICT: {result.verdict}")
    if result.confidence is not None:
        print(f"Confidence: {result.confidence}")
    print(f"Supporting evidence: {result.supporting_evidence_ids}")
    print(f"Contradicting evidence: {result.contradicting_evidence_ids}")
    print(f"\nExplanation:\n{result.explanation}")
    print("=" * 60)


if __name__ == "__main__":
    main()
