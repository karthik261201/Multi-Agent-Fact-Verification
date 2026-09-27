import json
import logging

import chainlit as cl
import requests
from pydantic import ValidationError

from agents.claim_agent import analyze_claim


logger = logging.getLogger(__name__)


# Convert a list of strings into a readable Markdown list.
def bullet_list(items):
    if not items:
        return "- None identified."
    return "\n".join(f"- {item}" for item in items)


# Runs when a new chat starts.
@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "# Claim Analysis\n\n"
            "Enter a factual claim to extract its entities, keywords, "
            "and search queries.\n\n"
            "**This interface analyzes claims; it does not verify "
            "whether they are true.**\n\n"
            "Each message is analyzed independently. If clarification "
            "is needed, send the complete revised claim."
        )
    ).send()


# Runs each time the user sends a message.
@cl.on_message
async def handle_message(message: cl.Message):
    claim = message.content

    # Check the input before calling the model.
    if not claim.strip():
        await cl.Message(content="Please enter a claim.").send()
        return

    if len(claim) > 2000:
        await cl.Message(
            content="Please keep your claim within 2,000 characters."
        ).send()
        return

    # Show progress while the local model works.
    reply = cl.Message(content="Analyzing your claim…")
    await reply.send()

    try:
        # Your agent uses blocking requests.post().
        # Run it in a worker thread so Chainlit can remain responsive.
        analysis = await cl.make_async(analyze_claim)(claim)

        if analysis.status == "needs_clarification":
            reply.content = (
                "## Clarification needed\n\n"
                + bullet_list(analysis.ambiguities)
                + "\n\nPlease send the complete claim again "
                "with the missing information included."
            )

        else:
            # Convert the internal analysis to your agreed output format.
            output = {
                "claim": analysis.original_claim,
                "search_queries": [
                    search.query for search in analysis.search_queries
                ],
                "entities": analysis.entities,
                "keywords": analysis.keywords,
            }

            formatted_json = json.dumps(
                output,
                indent=2,
                ensure_ascii=False,
            )

            reply.content = (
                "## Claim analysis\n\n"
                "**Status:** Ready for evidence retrieval "
                "— not verified as true.\n\n"
                "### Entities\n"
                + bullet_list(output["entities"])
                + "\n\n### Keywords\n"
                + bullet_list(output["keywords"])
                + "\n\n### Search queries\n"
                + bullet_list(output["search_queries"])
            )

            # Display any remaining notes even when the claim is ready.
            if analysis.ambiguities:
                reply.content += (
                    "\n\n### Interpretation notes\n"
                    + bullet_list(analysis.ambiguities)
                )

            # A separate panel displays the simplified JSON.
            reply.content += "\n\nOpen **Analysis JSON** to inspect the data."
            reply.elements = [
                cl.Text(
                    name="Analysis JSON",
                    content=formatted_json,
                    language="json",
                    display="side",
                )
            ]

    except requests.exceptions.Timeout:
        reply.content = (
            "The local model took too long to respond. "
            "Try a shorter claim and close unnecessary applications."
        )

    except requests.exceptions.RequestException:
        reply.content = (
            "The Ollama request failed. Check that Ollama is running "
            "and qwen2.5:3b is installed."
        )

    except ValidationError:
        reply.content = (
            "The model returned an analysis that did not match "
            "the required structure. Please try again."
        )

    except ValueError as error:
        reply.content = f"Analysis could not be completed: {error}"

    except Exception:
        # Keep technical details in the terminal for debugging.
        logger.exception("Unexpected error during claim analysis")
        reply.content = (
            "An unexpected error occurred. "
            "Check the terminal for details."
        )

    # Replace the progress message with the result or error.
    await reply.update()