# src/orchestrator/pipeline.py

from src.agents.claim_agent import analyze_claim
from src.agents.evidence_agent import retrieve_evidence
from src.agents.verification_agent import (prepare_evidence, call_verification_model)


def run_pipeline(user_claim: str):
    """
    Run the fact-verification pipeline.

    Current flow:
        1. Receive user's claim
        2. Claim Agent analyzes the claim
        3. Extract search queries from Claim Agent output
        4. Evidence Agent searches for evidence
        5. Return claim analysis and retrieved evidence

    Verification Agent will be added later.
    """

    # =========================================================
    # STEP 1: CLAIM ANALYSIS AGENT
    # =========================================================

    print("\n========== CLAIM ANALYSIS ==========")

    # Send the user's claim to the Claim Agent.
    analysis = analyze_claim(user_claim)

    # Display Claim Agent output.
    print("\nOriginal Claim:")
    print(analysis.original_claim)

    print("\nStatus:")
    print(analysis.status)

    print("\nEntities:")
    for entity in analysis.entities:
        print("-", entity)

    print("\nKeywords:")
    for keyword in analysis.keywords:
        print("-", keyword)

    print("\nSubclaims:")
    for subclaim in analysis.subclaims:
        print(f"{subclaim.id}: {subclaim.text}")

    print("\nSearch Queries:")
    for search in analysis.search_queries:
        print(f"{search.subclaim_id}: {search.query}")
    
    # =========================================================
    # CHECK WHETHER CLAIM IS READY
    # =========================================================

    # If the Claim Agent says more information is needed,
    # we should not continue to evidence retrieval.
    if analysis.status == "needs_clarification":
        print("\n========== CLARIFICATION REQUIRED ==========")

        for ambiguity in analysis.ambiguities:
            print("-", ambiguity)

        return {
            "analysis": analysis,
            "evidence_result": None
        }

    # =========================================================
    # STEP 2: PREPARE SEARCH QUERIES
    # =========================================================

    # Claim Agent returns SearchQuery objects:
    #
    # SearchQuery(
    #     subclaim_id="c1",
    #     query="Chandrayaan-3 launch date"
    # )
    #
    # Evidence Agent currently expects a list of strings.
    #
    # Therefore we extract only the query text.

    search_queries = [ search.query for search in analysis.search_queries ]

    # =========================================================
    # STEP 3: EVIDENCE RETRIEVAL AGENT
    # =========================================================

    print("\n========== EVIDENCE RETRIEVAL ==========")

    evidence_result = retrieve_evidence(
        claim=analysis.original_claim,
        search_queries=search_queries
    )

    # =========================================================
    # STEP 4: DISPLAY RETRIEVED EVIDENCE
    # =========================================================

    print("\n========== TOP EVIDENCE ==========")

    if not evidence_result["evidence"]:
        print("\nNo evidence was retrieved.")

    else:
        for index, item in enumerate(evidence_result["evidence"], start=1):
            print(f"\nEvidence {index}")
            print("Relevance:",round(item["relevance_score"], 3))
            print("Title:", item["title"])
            print("URL:", item["url"])
            print("Text:", item["text"])
    
    # =========================================================
    # STEP 5: PREPARE EVIDENCE FOR VERIFICATION
    # =========================================================

    print("\n========== PREPARING VERIFICATION INPUT ==========")

    # Convert Evidence Agent output into the format
    # expected by the Verification Agent.
    verification_evidence = prepare_evidence(evidence_result["evidence"])

    print(
        f"Prepared {len(verification_evidence)} "
        "evidence item(s) for verification."
    )
    
    # =========================================================
    # STEP 6: VERIFICATION AGENT
    # =========================================================

    print("\n========== VERIFICATION ==========")

    verification_result = call_verification_model(
        claim=analysis.original_claim,
        evidence=verification_evidence
    )

    # =========================================================
    # STEP 7: DISPLAY FINAL RESULT
    # =========================================================

    print("\n========== FINAL RESULT ==========")

    print("Verdict:", verification_result.verdict)

    print("Confidence:", verification_result.confidence)

    print("Supporting Evidence:", verification_result.supporting_evidence_ids)

    print("Contradicting Evidence:", verification_result.contradicting_evidence_ids)

    print("Explanation:", verification_result.explanation)

    # =========================================================
    # RETURN PIPELINE RESULTS
    # =========================================================

    # Returning both results will allow us to pass them
    # to the Verification Agent in the next integration step.

    return {
        "analysis": analysis,
        "evidence_result": evidence_result,
        "verification_result": verification_result
    }

if __name__ == "__main__":

    # Take the claim from the user.
    user_claim = input("Enter a claim: ")

    # Start the pipeline.
    run_pipeline(user_claim)