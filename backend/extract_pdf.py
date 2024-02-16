#!/usr/bin/python3
# This script extracts images and text from a PDF file.
# Usage:
# python extract_pdf.py --pdf your_pdf_file.pdf

import re
import fitz  # PyMuPDF
import os
import argparse


def extract_images_and_text(pdf_path, output_folder, inject_tokens=False):
    file_name = os.path.basename(pdf_path)

    # Create the output folder if it doesn't exist
    pdf_path_base_path = os.path.dirname(pdf_path)
    output_folder = os.path.join(pdf_path_base_path, output_folder)
    if not os.path.exists(output_folder):
        print(f"Creating output folder: {output_folder}")
        os.makedirs(output_folder)

    # Open the PDF file
    all_text_file = open(f"{output_folder}/{file_name}.txt", "w")
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Extract text and save to a file
            text = page.get_text()
            all_text_file.write(f"<PAGE {page_num + 1}/>\n")
            if inject_tokens:
                indexed_sentences = extract_segments(text)
                text = inject_indices(indexed_sentences, page_num + 1)
            all_text_file.write(text)

            # with open(f"{output_folder}/{page_num + 1}.txt", "w") as text_file:
            #     text_file.write(f"<PAGE {page_num + 1}/>\n")
            #     text_file.write(text)

            # Extract images
            image_list = page.get_images(full=True)
            for i, img in enumerate(image_list, start=1):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Save the image
                with open(f"{output_folder}/{page_num + 1}-{i}.jpg", "wb") as img_file:
                    img_file.write(image_bytes)
    all_text_file.close()
    print(f"Text and images extracted from {pdf_path} to {output_folder}")


def extract_from_folder(folder_path, output_folder, inject_tokens=False):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create an images subfolder
    images_folder = os.path.join(output_folder, "images")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            extract_images_and_text(pdf_path, output_folder)


def extract_segments(text: str) -> list[str]:
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


def inject_indices(indexed_sentences: list[str], page: int) -> str:
    # Initialize an empty string to hold the result
    combined_with_indices = ""

    # Loop through the list of indexed sentences
    for i, sentence in enumerate(indexed_sentences, start=1):
        # Append the sentence with its preceding index
        # Assuming <i> is the tag before the sentence and <i+1> is the tag after the sentence
        # For the last sentence, there's no next sentence, so just use <i>
        combined_with_indices += f"<P{page}S{i}/>{sentence}"

    return combined_with_indices


def main():
    parser = argparse.ArgumentParser(description="Extract images and text from a PDF.")
    parser.add_argument(
        "--folder", type=str, help="Path to the folder containing PDF files"
    )
    parser.add_argument("--pdf", type=str, help="Path to the PDF file")
    parser.add_argument(
        "--outfolder", type=str, default="outfolder", help="Path to the output folder"
    )
    parser.add_argument(
        "--inject_tokens", action="store_true", help="Inject tokens into the text"
    )
    args = parser.parse_args()

    if args.inject_tokens:
        print("Injecting tokens into the extracted text.")

    if args.folder:
        extract_from_folder(args.folder, args.outfolder, args.inject_tokens)
    elif args.pdf:
        extract_images_and_text(args.pdf, args.outfolder, args.inject_tokens)


if __name__ == "__main__":
    main()
