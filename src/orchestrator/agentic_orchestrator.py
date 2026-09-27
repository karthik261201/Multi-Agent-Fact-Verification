"""
Agentic Orchestrator
--------------------

Controls the multi-agent fact-verification workflow.

Unlike pipeline.py, which follows a fixed sequence,
this orchestrator can make decisions based on agent outputs.

Version 1:
    - Run Claim Agent
    - Run Evidence Agent
    - Run Verification Agent
    - If verdict is SUPPORTED or REFUTED -> stop
    - If verdict is INSUFFICIENT -> retry evidence retrieval
    - Stop after a maximum number of attempts

Later:
    - Generate improved search queries for retries
    - Evaluate evidence sufficiency before verification
    - Preserve subclaim-to-evidence mapping
    - Add more intelligent orchestration decisions
"""

from src.agents.claim_agent import analyze_claim
from src.agents.evidence_agent import retrieve_evidence
from src.agents.verification_agent import (prepare_evidence, call_verification_model)


# Maximum number of evidence retrieval attempts.
# This prevents the orchestrator from entering an infinite loop.
MAX_RETRIES = 2


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
        print(f"{subclaim.id}: {subclaim.text}")

    print("\nSearch Queries:")
    for search in analysis.search_queries:
        print(f"{search.subclaim_id}: {search.query}")

    # ========================================================
    # STEP 2: HANDLE AMBIGUOUS CLAIMS
    # ========================================================

    # If Agent 1 cannot clearly understand the claim,
    # the orchestrator should not continue blindly.
    if analysis.status == "needs_clarification":

        print("\n========== ORCHESTRATOR DECISION ==========")
        print("Claim requires clarification.")

        print("\nAmbiguities:")
        for ambiguity in analysis.ambiguities:
            print("-", ambiguity)

        return {
            "status": "needs_clarification",
            "analysis": analysis,
            "verification_result": None
        }

    # ========================================================
    # STEP 3: PREPARE INITIAL SEARCH QUERIES
    # ========================================================

    search_queries = [search.query for search in analysis.search_queries]

    # Keep track of the latest results.
    evidence_result = None
    verification_result = None

    # ========================================================
    # STEP 4: AGENTIC RETRY LOOP
    # ========================================================

    # Attempt 1 = initial retrieval
    # Attempt 2 = retry if evidence was insufficient
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
            search_queries=search_queries
        )

        evidence_count = len(evidence_result["evidence"])

        print(
            f"\nRetrieved {evidence_count} "
            "evidence item(s)."
        )

        # ----------------------------------------------------
        # PREPARE EVIDENCE FOR AGENT 3
        # ----------------------------------------------------

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

        # If the evidence clearly supports or refutes
        # the claim, verification is complete.
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

                # IMPORTANT:
                # Version 1 uses the same search queries.
                #
                # In the next version, the orchestrator
                # will generate improved/reformulated queries.

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

    # Return everything so that a future UI can consume it.
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