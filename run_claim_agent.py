import requests
from pydantic import ValidationError

from agents.claim_agent import analyze_claim


def main():
   
    claim = input("Enter a claim: ")

    print("\nAnalyzing claim...")

    try:
        
        analysis = analyze_claim(claim)

        print("\nClaim analysis:")
        print(analysis.model_dump_json(indent=2))

    except requests.exceptions.RequestException as error:
        print(f"\nOllama request failed: {error}")
        print("Check that Ollama is running and the model is available.")

    except ValidationError as error:
        print(f"\nThe model's response did not match our schema:\n{error}")

    except ValueError as error:
        print(f"\nAnalysis error: {error}")

    except (KeyError, TypeError):
        print("\nOllama returned an unexpected response structure.")


# Run the terminal interface only when this file is executed directly.
if __name__ == "__main__":
    main()