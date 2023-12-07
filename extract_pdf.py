#!/usr/bin/python3
# This script extracts images and text from a PDF file.
# Usage:
# python extract_pdf.py --pdf your_pdf_file.pdf

import fitz  # PyMuPDF
import os
import argparse

def extract_images_and_text(pdf_path, output_folder='output'):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the PDF file
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text and save to a file
            text = page.get_text()
            with open(f'{output_folder}/{page_num + 1}.txt', 'w') as text_file:
                text_file.write(text)

            # Extract images
            image_list = page.get_images(full=True)
            for i, img in enumerate(image_list, start=1):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Save the image
                with open(f'{output_folder}/{page_num + 1}-{i}.jpg', 'wb') as img_file:
                    img_file.write(image_bytes)

def main():
    parser = argparse.ArgumentParser(description='Extract images and text from a PDF.')
    parser.add_argument('--pdf', type=str, required=True, help='Path to the PDF file')
    parser.add_argument('--outfolder', type=str, help='Path to the output folder')
    args = parser.parse_args()

    extract_images_and_text(args.pdf, args.outfolder)

if __name__ == "__main__":
    main()
