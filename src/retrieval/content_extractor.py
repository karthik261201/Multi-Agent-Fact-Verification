# extract text from url

from trafilatura import fetch_url, extract

def extract_content(url):
    """
    Download a webpage and extract its main readable text.

    Parameters: url (str): URL of the webpage.

    Returns:
        str | None:
            Clean webpage text if extraction succeeds.
            None if the webpage cannot be downloaded
            or useful text cannot be extracted.
    """

    try:
        # Download the webpage.
        downloaded_page = fetch_url(url)

        # Some websites may block automated access.
        # In that case Trafilatura may return None.
        if downloaded_page is None:
            return None

        # Extract the useful textual content.
        # We don't need comments or tables for the
        # first version of our evidence system.
        text = extract(downloaded_page, include_comments=False, include_tables=False)

        return text

    except Exception as error:

        # A single failed website should NOT crash
        # the complete Evidence Agent.
        print(f"Could not extract {url}")
        print("Reason:", error)

        return None


if __name__ == "__main__":

    # Replace this with one URL returned by web_search.py.
    test_url = "https://science.nasa.gov/mission/chandrayaan-3"

    content = extract_content(test_url)

    if content:
        # Only print the first 2000 characters while testing.
        print(content[:2000])

    else:
        print("Could not extract content.")