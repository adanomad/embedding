import argparse
import os
import re
import time
import concurrent.futures
from dotenv import load_dotenv
import openai
from openai import OpenAI
import json
import csv
import pandas as pd
from typing import Optional
from pydantic import BaseModel
from typing import List, Dict

MODEL_NAME = "gpt-4-1106-preview"
# MODEL_NAME = "gpt-3.5-turbo"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No OpenAI API key found in environment variables")
openai_client = OpenAI(api_key=api_key)


# Function to read and split the text file by page
def read_and_split_file(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        content = file.read()
    # Replace newlines with spaces
    content = content.replace("\n", " ")
    # Use a split operation that retains the <PAGE prefix by using a regex split
    parts = re.split(r"(\<PAGE \d+.\>)", content)  # Split and keep <PAGE x>
    pages = []
    # Skip the first element if it's empty and start re-constructing pages with their <PAGE x> tags
    for i in range(1, len(parts), 2):
        pages.append(parts[i] + parts[i + 1])

    return pages


# Function to read and prepare the prompt
def prepare_prompt(page_text: str, prompt_path: str, page_number: int) -> str:
    with open(prompt_path, "r") as file:
        prompt = file.read()
    TO_REPLACE = "{{.dataframe}}"
    # raise error if the prompt does not contain the placeholder
    if TO_REPLACE not in prompt:
        raise ValueError(f"Prompt file {prompt_path} does not contain {TO_REPLACE}")
    gpt_input = prompt.replace(TO_REPLACE, page_text)
    # Write the prompt to file
    with open(f"{MODEL_NAME}.page{page_number + 1}.in.txt", "w") as file:
        file.write(gpt_input)
    return gpt_input


def send_to_chatgpt(input: str, page_number: int) -> dict:
    print(f"Sending to ChatGPT for page {page_number}...")
    start = time.time()
    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"content": input, "role": "user"}],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if content is None:
        print(f"No content error processing page: {input}")
        raise Exception(f"No content error processing page {page_number}")
    json_response = json.loads(content)
    print(f"Processed page {page_number} in {time.time() - start:.2f} seconds")

    # Write the response to file
    with open(f"page{page_number + 1}.{MODEL_NAME}.out.json", "w") as file:
        file.write(json.dumps(json_response, indent=2))
    print(f"Written response to page{page_number + 1}.prompt.out.txt")
    # Metrics (input length, response length, time taken, model used, page number)
    print(
        f"Metrics: Input length: {len(input)}, Response length: {len(content)}, Time taken: {time.time() - start:.2f} seconds, Model: {MODEL_NAME}, Page number: {page_number + 1}"
    )
    collect_metrics_to_csv(
        {
            "input_length": len(input),
            "response_length": len(content),
            "time_taken": time.time() - start,
            "model": MODEL_NAME,
            "page_number": page_number + 1,
        },
        "metrics.csv",
    )
    return json_response


def collect_metrics_to_csv(metric: Dict[str, str | int | float], csv_file: str):
    if not os.path.exists(csv_file):
        with open(csv_file, "w") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "input_length",
                    "response_length",
                    "time_taken",
                    "model",
                    "page_number",
                ]
            )
    with open(csv_file, "a") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                metric["input_length"],
                metric["response_length"],
                metric["time_taken"],
                metric["model"],
                metric["page_number"],
            ]
        )


# Main function to process the file and prompt
def process_file_and_prompt(txt_file: str, prompt_template: str):
    t0 = time.time()
    responses = []
    pages = read_and_split_file(txt_file)
    print(
        f"Read {len(pages)} pages from {txt_file} in {(time.time() - t0):.2f} seconds"
    )
    t1 = time.time()

    # Note PAGE_LIMIT is for testing purposes, so we don't have to wait for all pages to process before checking the results
    PAGE_LIMIT = 4
    for idx, page in enumerate(pages[:PAGE_LIMIT]):
        print(f"Processing page {pages.index(page) + 1} of {len(pages)}")
        if page == "":
            continue
        input = prepare_prompt(page, prompt_template, idx)

        print(f"Written prompt to page{idx + 1}.prompt.in.txt")
        response = send_to_chatgpt(input, idx + 1)
        responses.append(response)

    # 106 pages in 296 seconds (2.8 seconds per page)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     futures = []
    #     for idx, page in enumerate(pages):
    #         if page == "":
    #             continue
    #         input = prepare_prompt(page, prompt_template, idx)
    #         futures.append(executor.submit(send_to_chatgpt, input, idx + 1))
    #     for future in concurrent.futures.as_completed(futures):
    #         response = future.result()
    #         responses.append(response)

    print(f"Processed {len(pages)} pages in {time.time() - t1} seconds")
    return responses


def prompt_2(template: str, question: str, responses: pd.DataFrame) -> str:
    prompt = template.replace("[the_question]", question)
    prompt += (
        "---\n"
        + "page_number\tsummary\t"
        + "\t".join(responses.columns.difference(["page_number", "summary"]))
        + "\n"
    )
    for _, row in responses.iterrows():
        summary = row["summary"] if pd.notna(row["summary"]) else "No summary provided."
        page_info = f"{row['page_number']}\t{summary}"
        for topic in responses.columns.difference(["page_number", "summary"]):
            page_info += (
                f"\t{row[topic] if pd.notna(row[topic]) else 'No data provided.'}"
            )
        prompt += page_info + "\n"
    prompt += "---"

    return prompt


def part2_chatgpt(index: int, prompt: str) -> str:
    try:
        response = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"content": prompt, "role": "user"}],
        )
        content = response.choices[0].message.content
        if content is None:
            print(f"No content error processing question {index}: {response}")
            return ""
        return content

    except openai.OpenAIError as e:
        # Handle any errors from the OpenAI API
        return f"Error processing question {index}: {str(e)}"


# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--txtfile",
        type=str,
        default="../data/davebuster/DAVEBUSTER'SENTERTAINMENTINC_20220629_8-K_EX-101_CreditLoanAgreement.PDF.txt",
    )
    parser.add_argument("--prompt_1", type=str, default="credit.pass1.prompt.txt")
    parser.add_argument("--prompt_2", type=str, default="credit.pass2.prompt.txt")
    parser.add_argument("--csv", type=str, default="DAVEBUSTER.csv")
    args = parser.parse_args()

    if args.step == 1:
        print("Step 1")
        data = process_file_and_prompt(args.txtfile, args.prompt_1)
        print(f"Processed {len(data)} pages")
        # Write data to json file
        outfile = "output.json"
        with open(outfile, "w") as file:
            json.dump(data, file)
        print(f"Written data to {outfile}")

    elif args.step == 2:
        print("Step 2")
        df = pd.read_csv(args.csv)
        print(f"Read {len(df)} rows from {args.csv}")
        dataframes_dict = {}
        # make topic columns without the page_number, summary, original_page_text, and embedding columns
        topic_columns = set(df.columns.tolist()) - set(
            ["page_number", "summary", "original_page_text", "embedding"]
        )
        print(f"Topic columns: {topic_columns}")
