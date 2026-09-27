# break long pages into passages

def chunk_text(text, chunk_size=120):
    """
    Split a large webpage into smaller passages.

    Parameters:
        text (str): Extracted webpage content.

        chunk_size (int): Approximate number of words per passage.

    Returns:
        list[str]:Smaller text passages.
    """

    # Split the complete text into individual words.
    words = text.split()

    chunks = []

    # Move through the words chunk_size words at a time.
    for start in range(0, len(words), chunk_size):

        end = start + chunk_size

        chunk_words = words[start:end]

        # Convert the list of words back into text.
        chunk = " ".join(chunk_words)

        # Ignore extremely small/empty chunks.
        if chunk.strip():
            chunks.append(chunk)

    return chunks