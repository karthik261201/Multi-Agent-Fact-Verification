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


if __name__ == "__main__":

    test_text = """
    Chandrayaan-3

                                                        Active Mission


Chandrayaan-3 is an Indian Space Research Organization mission that landed near the south pole of the Moon on Aug. 23, 2023. The mission includes a lander and a rover. India plans to demonstrate end-to-end landing and roving capabilities.
Type
Launch
Target
Objective
Webb is the premier observatory of the next decade, serving thousands of astronomers worldwide. It studies every phase in the…
This rover and its aerial sidekick were assigned to study the geology of Mars and seek signs of ancient microbial…
On a mission to “touch the Sun,” NASA's Parker Solar Probe became the first spacecraft to fly through the corona…
NASA’s Juno spacecraft entered orbit around Jupiter in 2016, the first explorer to peer below the planet's dense clouds to
    """

    chunks = chunk_text(test_text, chunk_size=10)

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk)