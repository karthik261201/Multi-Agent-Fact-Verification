"""
Agentic Orchestrator
--------------------

Controls the multi-agent fact-verification workflow.

Unlike pipeline.py, which follows a fixed sequence,
this orchestrator can make decisions based on agent outputs.

Current capabilities:
    - Run Claim Agent
    - Run Evidence Agent
    - Run Verification Agent
    - Stop when evidence supports/refutes the claim
    - Retry when evidence is insufficient
    - Generate improved search queries for retry
    - Preserve subclaim/query provenance during
      the initial evidence retrieval
"""

import json

from groq import Groq

from src.agents.claim_agent import analyze_claim
from src.agents.evidence_agent import retrieve_evidence
from src.agents.verification_agent import (prepare_evidence, call_verification_model)

# Maximum number of evidence retrieval attempts.
MAX_RETRIES = 2

# ============================================================
# RETRY QUERY GENERATION
# ============================================================

def generate_retry_queries(claim, previous_queries, verification_explanation):
    """
    Generate improved search queries when the previous
    evidence retrieval attempt was insufficient.

    Uses:
        1. Original claim
        2. Previous search queries
        3. Verification Agent's explanation

    to improve the next retrieval attempt.
    """

    client = Groq()

    prompt = f"""
        You are controlling the evidence retrieval strategy of a
        fact-verification system.

        The previous evidence retrieval attempt was insufficient.

        ORIGINAL CLAIM:
        {claim}

        PREVIOUS SEARCH QUERIES:
        {json.dumps(previous_queries, indent=2)}

        WHY THE EVIDENCE WAS INSUFFICIENT:
        {verification_explanation}

        Generate improved web search queries that are more specific
        to the original claim and address the missing evidence.

        Rules:
        - Generate 2 to 4 queries.
        - Preserve important entities, dates, events, and qualifiers from the original claim.
        - Do not assume the claim is true.
        - Queries must be neutral.
        - Do not simply repeat the previous queries.
        - Focus specifically on information missing from the previous evidence.
        - Return ONLY valid JSON.

        Required format:

        {{
            "queries": [
                "query 1",
                "query 2"
            ]
        }}
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=500,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Read the model response.
    text = response.choices[0].message.content.strip()

    # Remove Markdown code fences if the model adds them.
    if text.startswith("```"):
        text = text.strip("`")

        if text.startswith("json"):
            text = text[4:].strip()

    # Convert the JSON response into a Python dictionary.
    data = json.loads(text)

    new_queries = data.get("queries", [])

    # Remove empty queries.
    new_queries = [
        query.strip()
        for query in new_queries
        if query.strip()
    ]

    return new_queries

# ============================================================
# AGENTIC ORCHESTRATOR
# ============================================================

def run_agentic_orchestrator(user_claim: str):
    """
    Run the agentic fact-verification workflow.
    """

    print("\n==========================================")
    print("       AGENTIC FACT VERIFICATION")
    print("==========================================")

    # ========================================================
    # STEP 1: CLAIM ANALYSIS
    # ========================================================

    print("\n========== CLAIM AGENT ==========")

    analysis = analyze_claim(user_claim)

    print("Status:", analysis.status)

    print("\nSubclaims:")
    for subclaim in analysis.subclaims:
        print(
            f"{subclaim.id}: "
            f"{subclaim.text}"
        )

    print("\nSearch Queries:")
    for search in analysis.search_queries:
        print(
            f"{search.subclaim_id}: "
            f"{search.query}"
        )

    # ========================================================
    # STEP 2: HANDLE AMBIGUOUS CLAIMS
    # ========================================================

    if analysis.status == "needs_clarification":

        print(
            "\n========== ORCHESTRATOR DECISION =========="
        )

        print(
            "Claim requires clarification."
        )

        print("\nAmbiguities:")
        for ambiguity in analysis.ambiguities:
            print("-", ambiguity)

        return {
            "status": "needs_clarification",
            "analysis": analysis,
            "evidence_result": None,
            "verification_result": None
        }

    # ========================================================
    # STEP 3: PREPARE INITIAL SEARCH QUERIES
    # ========================================================

    # IMPORTANT:
    #
    # Previously we converted SearchQuery objects into strings.
    #
    # Now we preserve the complete objects so that the
    # Evidence Agent knows which query belongs to which
    # subclaim.
    search_queries = analysis.search_queries
    evidence_result = None
    verification_result = None

    # ========================================================
    # STEP 4: AGENTIC RETRY LOOP
    # ========================================================

    for attempt in range(1, MAX_RETRIES + 1):

        print(
            f"\n========== EVIDENCE ATTEMPT "
            f"{attempt}/{MAX_RETRIES} =========="
        )

        # ----------------------------------------------------
        # CALL EVIDENCE AGENT
        # ----------------------------------------------------

        evidence_result = retrieve_evidence(
            claim=analysis.original_claim,
            search_queries=search_queries,
            subclaims=analysis.subclaims
        )

        evidence_count = len(evidence_result["evidence"])

        print(
            f"\nRetrieved {evidence_count} "
            "evidence item(s)."
        )

        # ====================================================
        # DISPLAY EVIDENCE PROVENANCE
        # ====================================================

        # This allows us to verify that each evidence item
        # remembers which subclaim/query retrieved it.
        print("\nEvidence Provenance:")

        for index, item in enumerate(evidence_result["evidence"], start=1):
            print(
                f"[{index}] "
                f"Subclaim: "
                f"{item.get('subclaim_id')} | "
                f"Query: "
                f"{item.get('query')}"
            )

        # ====================================================
        # PREPARE EVIDENCE FOR VERIFICATION AGENT
        # ====================================================

        verification_evidence = prepare_evidence(evidence_result["evidence"])

        # ----------------------------------------------------
        # CALL VERIFICATION AGENT
        # ----------------------------------------------------

        print("\n========== VERIFICATION AGENT ==========")

        verification_result = call_verification_model(
            claim=analysis.original_claim,
            evidence=verification_evidence
        )

        print("Verdict:", verification_result.verdict)

        print("Confidence:", verification_result.confidence)

        print("Explanation:", verification_result.explanation)

        # ====================================================
        # STEP 5: ORCHESTRATOR DECISION
        # ====================================================

        print("\n========== ORCHESTRATOR DECISION ==========")

        # ----------------------------------------------------
        # SUFFICIENT EVIDENCE
        # ----------------------------------------------------
        if verification_result.verdict in ["SUPPORTED", "REFUTED"]:

            print(
                "Evidence is sufficient. "
                "Verification complete."
            )

            break

        # ----------------------------------------------------
        # INSUFFICIENT EVIDENCE
        # ----------------------------------------------------

        if verification_result.verdict == "INSUFFICIENT":

            # If another attempt is available, retry.
            if attempt < MAX_RETRIES:

                print(
                    "Evidence is insufficient."
                )

                print(
                    "Orchestrator decision: "
                    "retry evidence retrieval."
                )

                # --------------------------------------------
                # PREPARE PREVIOUS QUERIES
                # --------------------------------------------

                # search_queries may contain:
                #
                # - SearchQuery objects
                # - plain strings from an earlier retry
                #
                # Convert both forms into strings before
                # sending them to the retry-query generator.
                previous_query_strings = [
                    search.query
                    if hasattr(search, "query")
                    else search
                    for search in search_queries
                ]

                # --------------------------------------------
                # GENERATE IMPROVED QUERIES
                # --------------------------------------------

                new_queries = generate_retry_queries(
                    claim=analysis.original_claim,
                    previous_queries=(
                        previous_query_strings
                    ),
                    verification_explanation=(
                        verification_result.explanation
                    )
                )

                print("\nPrevious Search Queries:")
                for query in previous_query_strings:
                    print("-", query)

                print("\nImproved Search Queries:")
                for query in new_queries:
                    print("-", query)

                if new_queries:
                    search_queries = new_queries

            else:

                print(
                    "Evidence is still insufficient."
                )

                print(
                    "Maximum retrieval attempts reached."
                )

    # ========================================================
    # STEP 6: FINAL RESULT
    # ========================================================

    print("\n==========================================")
    print("              FINAL RESULT")
    print("==========================================")

    print("Verdict:", verification_result.verdict)

    print("Confidence:", verification_result.confidence)

    print("Supporting Evidence:",verification_result.supporting_evidence_ids)

    print("Contradicting Evidence:",verification_result.contradicting_evidence_ids)

    print("Explanation:",verification_result.explanation)

    # ========================================================
    # RETURN COMPLETE RESULT
    # ========================================================

    return {
        "status": "completed",
        "analysis": analysis,
        "evidence_result": evidence_result,
        "verification_result": verification_result
    }


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    user_claim = input("Enter a claim: ")

    run_agentic_orchestrator(user_claim)