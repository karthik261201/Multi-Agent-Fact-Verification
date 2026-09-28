# connecting all four retrieval components together

from src.retrieval.web_search import search_web
from src.retrieval.content_extractor import extract_content
from src.retrieval.text_chunker import chunk_text
from src.retrieval.evidence_ranker import rank_evidence


def retrieve_evidence(claim, search_queries, subclaims=None, max_results_per_query=3, top_k_per_subclaim=3):
    """
    Complete Evidence Retrieval Agent.

    The agent:
        1. Searches the web
        2. Downloads relevant webpages
        3. Extracts webpage content
        4. Splits content into passages
        5. Ranks passages against the claim
        6. Returns the strongest evidence

    Parameters:
        claim (str): Original claim entered by the user.

        search_queries (list[str]): Queries produced by the Claim Agent.

    Returns:
        list[dict]: Top evidence passages with source information.
    
    Retrieve and rank web evidence for a claim.

    search_queries can contain:

    1. SearchQuery objects from the Claim Agent:
       - subclaim_id
       - query

    2. Plain string queries:
       - Used by orchestrator-generated retry queries.

    Each retrieved evidence passage keeps track of:
       - which subclaim it belongs to
       - which search query retrieved it
       - source title
       - source URL
       - evidence text
       - relevance score
    """

    """
    Retrieve and rank web evidence.

    Initial Claim Agent queries contain subclaim IDs.
    When subclaim information is available, evidence is ranked
    separately for each subclaim.

    Plain string queries are still supported for orchestrator
    retry attempts.
    """

    # ========================================================
    # BUILD SUBCLAIM LOOKUP
    # ========================================================

    # Example:
    #
    # {
    #     "c1": "Chandrayaan-3 was launched by ISRO",
    #     "c2": "Chandrayaan-3 was launched in July 2023"
    # }
    subclaim_lookup = {}

    if subclaims:
        for subclaim in subclaims:
            subclaim_lookup[subclaim.id] = subclaim.text

    all_passages = []

    # ========================================================
    # PROCESS EACH SEARCH QUERY
    # ========================================================

    for search_item in search_queries:

        # ----------------------------------------------------
        # CASE 1:
        # SearchQuery object produced by Claim Agent
        # ----------------------------------------------------

        if hasattr(search_item, "query"):
            query = search_item.query
            subclaim_id = search_item.subclaim_id

        # ----------------------------------------------------
        # CASE 2:
        # Plain string query produced by orchestrator retry
        # ----------------------------------------------------

        else:
            query = search_item
            subclaim_id = None

        print(f"\nSearching: {query}")

        # ====================================================
        # WEB SEARCH
        # ====================================================

        search_results = search_web(query, max_results=max_results_per_query)

        # ====================================================
        # PROCESS SEARCH RESULTS
        # ====================================================

        for result in search_results:

            url = result["url"]

            print(f"Reading: {url}")

            # Extract useful text from the webpage.
            content = extract_content(url)

            # Skip pages where extraction failed.
            if not content:
                continue

            # Split large webpage text into smaller passages.
            chunks = chunk_text(content)

            # =================================================
            # STORE PASSAGES WITH PROVENANCE
            # =================================================

            for chunk in chunks:

                all_passages.append({
                    # Which Claim Agent subclaim caused
                    # this evidence to be retrieved.
                    "subclaim_id": subclaim_id,

                    # Exact search query that retrieved it.
                    "query": query,

                    # Evidence content.
                    "text": chunk,

                    # Source information.
                    "title": result["title"],
                    "url": url
                })
        
        # Check whether retrieved passages actually have
        # subclaim provenance.
        has_mapped_passages = any(passage.get("subclaim_id") is not None for passage in all_passages)

        # ========================================================
        # PER-SUBCLAIM RANKING
        # ========================================================

        final_evidence = []
        # If we have structured subclaim information,
        # rank evidence separately for each subclaim.
        if subclaim_lookup and has_mapped_passages:
            for subclaim_id, subclaim_text in (subclaim_lookup.items()):
                # Select only evidence retrieved for this subclaim.
                subclaim_passages = [passage for passage in all_passages if passage.get("subclaim_id") == subclaim_id]

                # No evidence was retrieved for this subclaim.
                if not subclaim_passages:
                    continue

                # IMPORTANT:
                #
                # Rank against the SUBCLAIM,
                # not the complete original claim.
                ranked = rank_evidence(subclaim_text, subclaim_passages,top_k=top_k_per_subclaim)
                final_evidence.extend(ranked)

        # ========================================================
        # FALLBACK FOR PLAIN STRING QUERIES
        # ========================================================

        else:
            # Retry queries currently do not have subclaim IDs.
            #
            # Therefore we temporarily keep the old global-ranking
            # behaviour for those queries.
            final_evidence = rank_evidence(claim, all_passages, top_k=top_k_per_subclaim)

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "claim": claim,
        "evidence": final_evidence
    }