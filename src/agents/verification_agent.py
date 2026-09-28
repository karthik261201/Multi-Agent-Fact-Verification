"""
Verification Agent
------------------

Receives:
    - The original claim
    - Evidence retrieved by the Evidence Agent

Returns:
    - Verdict: SUPPORTED / REFUTED / INSUFFICIENT
    - Confidence score
    - Supporting evidence IDs
    - Contradicting evidence IDs
    - Evidence-grounded explanation
"""

import json

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
    """
    Evidence passed from the Evidence Agent
    to the Verification Agent.

    subclaim_id preserves which subclaim this
    evidence was originally retrieved for.
    """

    id: int
    # Claim Agent subclaim this evidence belongs to.
    subclaim_id: Optional[str] = None
    # Search query that retrieved this evidence.
    query: str = ""
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

@dataclass
class SubclaimVerificationResult:
    """
    Stores the verification result for one individual subclaim.
    """
    subclaim_id: str
    subclaim_text: str
    verdict: str
    explanation: str
    supporting_evidence_ids: List[int] = field(default_factory=list)
    contradicting_evidence_ids: List[int] = field(default_factory=list)
    confidence: Optional[float] = None

@dataclass
class OverallVerificationResult:
    """
    Final verification result containing both:
    - individual subclaim results
    - overall claim verdict
    """
    verdict: str
    explanation: str
    subclaim_results: List[SubclaimVerificationResult] = field(default_factory=list)
    supporting_evidence_ids: List[int] = field(default_factory=list)
    contradicting_evidence_ids: List[int] = field(default_factory=list)
    confidence: Optional[float] = None

# ============================================================
# EVIDENCE CONVERSION
# ============================================================

def prepare_evidence(evidence_items):
    """
    Convert Evidence Agent dictionaries into Evidence
    objects used by the Verification Agent.

    Provenance information is preserved so that evidence
    can later be verified per subclaim.
    """

    evidence = []

    for index, item in enumerate(evidence_items, start=1):
        evidence.append(
            Evidence(
                id=index,
                subclaim_id=item.get("subclaim_id"),
                query=item.get("query", ""),
                title=item.get("title", ""),
                url=item.get("url", ""),
                relevance=item.get("relevance_score"),
                snippet=item.get("text", "")
            )
        )

    return evidence

# ============================================================
# VERIFICATION PROMPT
# ============================================================

SYSTEM_PROMPT = """
    You are the Verification Agent in a multi-agent fact-checking pipeline.

    You receive:
    1. A CLAIM
    2. Evidence retrieved by a separate Evidence Retrieval Agent

    Your job is ONLY to determine whether the provided evidence
    supports, refutes, or is insufficient to establish the claim.

    Rules:

    - Base your verdict strictly on the provided evidence.
    - Do not use outside knowledge to invent facts.
    - Do not search for additional information.

    Verdicts:

    SUPPORTED:
    The provided evidence clearly supports the claim.

    REFUTED:
    The provided evidence clearly contradicts the claim.

    INSUFFICIENT:
    The evidence is irrelevant, too weak, incomplete, or does not
    clearly establish whether the claim is correct.

    Additional rules:

    - Consider all provided evidence.
    - Note contradictions between evidence items.
    - Identify which evidence IDs support the claim.
    - Identify which evidence IDs contradict the claim.
    - Provide a short explanation grounded in the evidence.
    - Confidence must be between 0.0 and 1.0.

    Respond ONLY with valid JSON.

    Required format:

    {
        "verdict": "SUPPORTED",
        "confidence": 0.95,
        "supporting_evidence_ids": [1, 2],
        "contradicting_evidence_ids": [],
        "explanation": "Evidence [1] and [2] directly support the claim."
    }
"""

# ============================================================
# BUILD PROMPT FOR THE MODEL
# ============================================================

def build_user_prompt(claim, evidence):
    """
    Build the prompt containing the claim and
    retrieved evidence for the Verification Agent.
    """

    evidence_text = ""

    for item in evidence:
        evidence_text += f"""
            Evidence ID: {item.id}
            Subclaim ID: {item.subclaim_id}
            Search Query: {item.query}
            Relevance: {item.relevance}
            Title: {item.title}
            URL: {item.url}
            Evidence Text:{item.snippet}
            ----------------------------------------
        """

    return f"""
        CLAIM: {claim}
        EVIDENCE: {evidence_text}
    """

# ============================================================
# CALL VERIFICATION MODEL
# ============================================================

def call_verification_model(claim: str, evidence: List[Evidence]) -> VerificationResult:
    if not evidence:
        return VerificationResult(
            verdict="INSUFFICIENT",
            confidence=0.0,
            supporting_evidence_ids=[],
            contradicting_evidence_ids=[],
            explanation="No evidence was retrieved for the claim."
        )

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

def verify_subclaims(subclaims, evidence):
    """
    Verify each subclaim independently using only the
    evidence retrieved for that specific subclaim.

    This prevents evidence belonging to one subclaim from
    incorrectly influencing another subclaim.
    """

    results = []

    for subclaim in subclaims:

        # ====================================================
        # GET EVIDENCE FOR THIS SUBCLAIM
        # ====================================================

        subclaim_evidence = [item for item in evidence if item.subclaim_id == subclaim.id]

        print(
            f"\nVerifying {subclaim.id}: "
            f"{subclaim.text}"
        )

        print(
            f"Evidence available: "
            f"{len(subclaim_evidence)}"
        )

        # ====================================================
        # VERIFY THIS SUBCLAIM
        # ====================================================

        result = call_verification_model(claim=subclaim.text, evidence=subclaim_evidence)

        # ====================================================
        # STORE RESULT
        # ====================================================

        results.append(
            SubclaimVerificationResult(
                subclaim_id=subclaim.id,
                subclaim_text=subclaim.text,
                verdict=result.verdict,
                explanation=result.explanation,
                supporting_evidence_ids=(result.supporting_evidence_ids),
                contradicting_evidence_ids=(result.contradicting_evidence_ids),
                confidence=result.confidence
            )
        )

    return results

def aggregate_subclaim_results(subclaim_results):
    """
    Combine individual subclaim verdicts into the final claim-level verdict.

    Rules:

    1. If ANY subclaim is REFUTED: overall = REFUTED

    2. If ALL subclaims are SUPPORTED: overall = SUPPORTED

    3. Otherwise: overall = INSUFFICIENT

    The aggregation itself is deterministic and does not
    require another LLM call.
    """

    if not subclaim_results:
        return OverallVerificationResult(
            verdict="INSUFFICIENT",
            confidence=0.0,
            explanation=("No subclaims were available for verification."),
            subclaim_results=[]
        )

    verdicts = [result.verdict for result in subclaim_results]

    # ========================================================
    # DETERMINE OVERALL VERDICT
    # ========================================================

    if "REFUTED" in verdicts:
        overall_verdict = "REFUTED"

    elif all(verdict == "SUPPORTED" for verdict in verdicts):
        overall_verdict = "SUPPORTED"

    else:
        overall_verdict = "INSUFFICIENT"

    # ========================================================
    # COLLECT EVIDENCE IDS
    # ========================================================

    supporting_ids = sorted(
        {
            evidence_id
            for result in subclaim_results
            for evidence_id
            in result.supporting_evidence_ids
        }
    )

    contradicting_ids = sorted(
        {
            evidence_id
            for result in subclaim_results
            for evidence_id
            in result.contradicting_evidence_ids
        }
    )

    # ========================================================
    # CALCULATE SIMPLE OVERALL CONFIDENCE
    # ========================================================

    confidences = [
        result.confidence
        for result in subclaim_results
        if result.confidence is not None
    ]

    overall_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else None
    )

    # ========================================================
    # BUILD EXPLANATION
    # ========================================================

    explanation_parts = []

    for result in subclaim_results:
        explanation_parts.append(
            f"{result.subclaim_id} "
            f"({result.verdict}): "
            f"{result.explanation}"
        )

    overall_explanation = " ".join(explanation_parts)

    return OverallVerificationResult(
        verdict=overall_verdict,
        confidence=overall_confidence,
        explanation=overall_explanation,
        subclaim_results=subclaim_results,
        supporting_evidence_ids=supporting_ids,
        contradicting_evidence_ids=contradicting_ids
    )

def verify_claim_by_subclaims(subclaims, evidence):
    """
    Complete Verification Agent workflow:

    1. Verify every subclaim independently.
    2. Aggregate subclaim results.
    3. Return the final claim-level result.
    """

    subclaim_results = verify_subclaims(subclaims=subclaims, evidence=evidence)

    return aggregate_subclaim_results(subclaim_results)