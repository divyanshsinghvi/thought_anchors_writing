from typing import List

def split_solution_into_chunks(solution_text: str) -> List[str]:
    """
    Split a solution into chunks for rollout generation.

    Args:
        solution_text: The full solution text

    Returns:
        List of chunks
    """
    # First, remove the prompt part if present
    if "<think>" in solution_text:
        solution_text = solution_text.split("<think>")[1].strip()

    # Remove the closing tag if present
    if "</think>" in solution_text:
        solution_text = solution_text.split("</think>")[0].strip()

    # Define patterns for chunk boundaries
    sentence_ending_tokens = [".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]

    # Split the text into chunks
    chunks = []
    current_chunk = ""

    # Process the text character by character
    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]

        # Check for paragraph endings
        is_paragraph_end = False
        for pattern in paragraph_ending_patterns:
            if (
                i + len(pattern) <= len(solution_text)
                and solution_text[i : i + len(pattern)] == pattern
            ):
                is_paragraph_end = True
                break

        # Check for sentence endings followed by space or newline
        is_sentence_end = False
        if i < len(solution_text) - 1 and solution_text[i] in sentence_ending_tokens:
            next_char = solution_text[i + 1]
            if next_char == " " or next_char == "\n":
                is_sentence_end = True

        # If we found a boundary, add the chunk and reset
        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

        i += 1

    # # Add the last chunk if not empty
    # if current_chunk.strip():
    #     chunks.append(current_chunk.strip())
    #     chunk_idxs.append(len(solution_text) - 1)  # Add last index

    # Merge small chunks (less than 10 characters)
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 10:
            # If this is the last chunk, merge with previous chunk if possible
            if i == len(chunks) - 1:
                if i > 0:
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    chunks.pop(i)
            # Otherwise merge with the next chunk
            else:
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)
                # Don't increment i since we need to check the new merged chunk
            # If we're at the beginning and there's only one chunk, just keep it
            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1

    # chunk_boundaries = [(chunk_idxs[i], chunk_idxs[i + 1]) for i in range(len(chunk_idxs) - 1)]
    # chunk_boundaries.append((chunk_idxs[-1], len(solution_text)))

    # if get_idxs:
    #     return chunks, chunk_boundaries
    # else:
    return chunks