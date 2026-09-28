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
    You are the Verification Agent in an evidence-grounded
    fact-verification system.

    You receive:
    1. A claim or subclaim.
    2. Retrieved evidence passages.

    Your task is to determine whether the supplied evidence
    SUPPORTS, REFUTES, or is INSUFFICIENT to determine the claim.

    IMPORTANT GROUNDING RULES:

    - Judge ONLY from the supplied evidence.
    - Do NOT use outside knowledge.
    - Do NOT invent facts.
    - Do NOT search for additional information.
    - Every supporting or contradicting evidence ID must refer to evidence actually provided to you.

    VERDICT RULES:

    SUPPORTED: Use this only when the supplied evidence clearly supports the claim.

    REFUTED: Use REFUTED only when the supplied evidence directly
    establishes that the claim cannot be true.A different date, event, property, 
    person, organization, or value does NOT automatically contradict the claim unless
    the evidence establishes that the alternatives are mutually exclusive.

    INSUFFICIENT: Use this when the evidence is missing, weak, irrelevant, ambiguous, 
    incomplete, or does not clearly establish either support or contradiction.

    CRITICAL RULE ABOUT ABSENCE OF EVIDENCE:

    The absence of supporting evidence is NOT itself evidence
    that a claim is false.

    If the supplied evidence simply does not mention the claimed
    event, property, date, relationship, or fact, you MUST NOT
    treat that omission alone as contradiction.

    For REFUTED, there must be supplied evidence that directly
    contradicts the claim or establishes an incompatible fact.

    Do not infer contradiction merely because evidence provides
    a different date, value, event, or attribute.

    Before returning REFUTED, ask:

    "Can the claim and this evidence both reasonably be true?"

    If YES, the evidence is not a direct contradiction.

    If the evidence does not otherwise establish the claim,
    return INSUFFICIENT.

    Example:

    Claim: "The monument was painted in January 2025."

    Evidence: "The monument is scheduled to be repainted in 2026."

    This alone is NOT sufficient to refute the claim because
    painting in 2025 and repainting in 2026 could both occur.
    Return INSUFFICIENT unless the evidence establishes that
    no painting occurred in January 2025.

    Claim: "Organization A launched the spacecraft."

    Evidence: "The spacecraft was launched by Organization B."

    If the evidence clearly identifies Organization B as the
    launching organization for that specific launch, this is
    directly incompatible with Organization A being the launching
    organization and may be REFUTED.

    EVIDENCE ID RULES:

    - supporting_evidence_ids must contain only evidence that directly supports the claim.
    - contradicting_evidence_ids must contain only evidence that directly contradicts the claim.
    - Never invent an evidence ID.
    - The same evidence ID must not appear in both lists.

    CONFIDENCE:

    Return a confidence value between 0.0 and 1.0.

    OUTPUT:

    Return ONLY valid JSON in exactly this structure:

    {
        "verdict": "SUPPORTED",
        "confidence": 0.95,
        "supporting_evidence_ids": [1, 2],
        "contradicting_evidence_ids": [],
        "explanation": "Evidence [1] and [2] directly support the claim."
    }

    Allowed verdicts:
    SUPPORTED
    REFUTED
    INSUFFICIENT
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

def validate_verification_output(data,evidence):
    """
    Deterministically validate the Verification Agent's
    LLM-generated JSON before accepting it.

    This protects the pipeline from:
    - invalid verdicts
    - invalid confidence values
    - hallucinated evidence IDs
    - duplicate IDs
    - evidence appearing as both support and contradiction
    - missing explanations
    """

    # ========================================================
    # VALID VERDICTS
    # ========================================================

    valid_verdicts = {
        "SUPPORTED",
        "REFUTED",
        "INSUFFICIENT"
    }

    verdict = data.get("verdict")

    if verdict not in valid_verdicts:
        raise ValueError(
            f"Invalid verification verdict: {verdict}"
        )

    # ========================================================
    # VALIDATE CONFIDENCE
    # ========================================================

    confidence = data.get("confidence")

    if not isinstance(confidence,(int, float)):
        raise ValueError(
            "Verification confidence must be numeric."
        )

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "Verification confidence must be between "
            "0.0 and 1.0."
        )

    # ========================================================
    # GET EVIDENCE IDS
    # ========================================================

    supporting_ids = data.get("supporting_evidence_ids",[])

    contradicting_ids = data.get("contradicting_evidence_ids",[])

    if not isinstance(supporting_ids, list):
        raise ValueError(
            "supporting_evidence_ids must be a list."
        )

    if not isinstance(contradicting_ids, list):
        raise ValueError(
            "contradicting_evidence_ids must be a list."
        )

    # ========================================================
    # MAKE SURE IDS ARE INTEGERS
    # ========================================================

    if not all(isinstance(item, int) for item in supporting_ids):
        raise ValueError(
            "All supporting evidence IDs must be integers."
        )

    if not all(isinstance(item, int) for item in contradicting_ids):
        raise ValueError(
            "All contradicting evidence IDs must be integers."
        )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    supporting_ids = list(dict.fromkeys(supporting_ids))

    contradicting_ids = list(dict.fromkeys(contradicting_ids))

    # ========================================================
    # CHECK THAT EVIDENCE IDS ACTUALLY EXIST
    # ========================================================

    valid_evidence_ids = {item.id for item in evidence}

    invalid_supporting = (set(supporting_ids) - valid_evidence_ids)

    invalid_contradicting = (set(contradicting_ids) - valid_evidence_ids)

    if invalid_supporting:
        raise ValueError(
            "Invalid supporting evidence IDs: "
            f"{sorted(invalid_supporting)}"
        )

    if invalid_contradicting:
        raise ValueError(
            "Invalid contradicting evidence IDs: "
            f"{sorted(invalid_contradicting)}"
        )

    # ========================================================
    # SAME EVIDENCE CANNOT SUPPORT AND CONTRADICT
    # ========================================================

    overlapping_ids = (set(supporting_ids) & set(contradicting_ids))

    if overlapping_ids:
        raise ValueError(
            "Evidence IDs cannot be both supporting "
            "and contradicting: "
            f"{sorted(overlapping_ids)}"
        )

    # ========================================================
    # VERDICT ↔ EVIDENCE CONSISTENCY
    # ========================================================

    if (verdict == "SUPPORTED" and not supporting_ids):
        raise ValueError(
            "SUPPORTED verdict requires at least one "
            "supporting evidence ID."
        )

    if (verdict == "REFUTED" and not contradicting_ids):
        raise ValueError(
            "REFUTED verdict requires at least one "
            "contradicting evidence ID."
        )

    # ========================================================
    # EXPLANATION
    # ========================================================

    explanation = data.get("explanation", "")

    if ( not isinstance(explanation, str)or not explanation.strip()):
        raise ValueError("Verification explanation cannot be empty.")

    # ========================================================
    # RETURN CLEAN DATA
    # ========================================================

    return {
        "verdict": verdict,
        "confidence": float(confidence),
        "supporting_evidence_ids": supporting_ids,
        "contradicting_evidence_ids": contradicting_ids,
        "explanation": explanation.strip()
    }

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

    validated_data = validate_verification_output(data=data, evidence=evidence)
    
    return VerificationResult(
        verdict=validated_data["verdict"],
        explanation=validated_data["explanation"],
        supporting_evidence_ids=validated_data.get("supporting_evidence_ids", []),
        contradicting_evidence_ids=validated_data.get("contradicting_evidence_ids", []),
        confidence=validated_data.get("confidence"),
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