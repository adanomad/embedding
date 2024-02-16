import re


def extract_information_segments(text):
    # Define patterns for identifying key segments; these can be adjusted or expanded based on the document's structure
    patterns = [
        # Match Roman numerals in parentheses. This pattern is simplified and might not cover all edge cases.
        r"\((?=[MDCLXVI])(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))\)",
        # Alphabetical items in parentheses (unchanged, as it suits the requirement)
        r"\([a-z]\)",
        # Adding a pattern for semicolons as a potential separator for items within the same sentence
        r";",
        # Example to match specific keywords that might indicate the start of a new clause or section.
        # Adjust the keyword list based on your text's characteristics.
        # This is a basic example and might need refinement.
        r"\b(including|without limitation|such as|e.g.,)\b",
    ]

    segment_indices = []
    for pattern in patterns:
        # Find all matches of the pattern in the text
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Store the match position (start index)
            segment_indices.append(match.start())

    # Sort indices and extract segments based on these indices
    segment_indices = sorted(list(set(segment_indices)))  # Remove duplicates and sort
    segments = []
    for i in range(len(segment_indices) - 1):
        # Extract text segments based on the indices
        segments.append(text[segment_indices[i] : segment_indices[i + 1]].strip())

    # Add the final segment if not captured
    if segment_indices:
        segments.append(text[segment_indices[-1] :].strip())

    return segments


def inject_indices_and_combine(indexed_sentences):
    # Initialize an empty string to hold the result
    combined_with_indices = ""

    # Loop through the list of indexed sentences
    for i, sentence in enumerate(indexed_sentences, start=1):
        # Append the sentence with its preceding index
        # Assuming <i> is the tag before the sentence and <i+1> is the tag after the sentence
        # For the last sentence, there's no next sentence, so just use <i>
        combined_with_indices += f"<SEG {i}>{sentence}"

    return combined_with_indices


# Example usage
if __name__ == "__main__":
    text = "This is a sample text. (I) This is another sentence. (II) This is a third sentence. (III) And so on."
    indexed_sentences = extract_information_segments(text)
    print(indexed_sentences)

    combined_text = inject_indices_and_combine(indexed_sentences)
    print(combined_text)
