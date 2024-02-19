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
MAX_TEXT_LENGTH = 4096 * 4  # 16KB

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No OpenAI API key found in environment variables")
openai_client = OpenAI(api_key=api_key)


def read_and_split_file(file_path: str) -> List[str]:
    pages = []
    with open(file_path, "r") as file:
        current_page = ""
        for line in file:
            if len(line) + len(current_page) > MAX_TEXT_LENGTH:
                pages.append(current_page)
                current_page = line
            else:
                current_page += " " + line
    # Don't forget to add the last page if it's not empty
    if current_page:
        pages.append(current_page)
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

    # # Note PAGE_LIMIT is for testing purposes, so we don't have to wait for all pages to process before checking the results
    # PAGE_LIMIT = 4
    # for idx, page in enumerate(pages[:PAGE_LIMIT]):
    #     print(f"Processing page {pages.index(page) + 1} of {len(pages)}")
    #     if page == "":
    #         continue
    #     input = prepare_prompt(page, prompt_template, idx)

    #     print(f"Written prompt to page{idx + 1}.prompt.in.txt")
    #     response = send_to_chatgpt(input, idx + 1)
    #     responses.append(response)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for idx, page in enumerate(pages):
            if page == "":
                continue
            input = prepare_prompt(page, prompt_template, idx)
            futures.append(executor.submit(send_to_chatgpt, input, idx + 1))
        for future in concurrent.futures.as_completed(futures):
            response = future.result()
            responses.append(response)

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


def read_and_clean_data(file_path):
    """Read JSON data from a file and remove specified columns."""
    with open(file_path, "r") as file:
        data = json.load(file)
    cleaned_data = [
        {
            k: v
            for k, v in page.items()
            if k not in ["page_number", "summary", "original_page_text", "embedding"]
        }
        for page in data
    ]
    return pd.DataFrame(cleaned_data)


def filter_data_for_sql(df):
    """Filter data to prepare rows for SQL insertion."""
    to_sql_table = []
    for topic in df.columns:
        filtered_df = df[df[topic].notna()][[topic]]
        print(f"Transformed {len(filtered_df)} rows for {topic}")
        for _, row in filtered_df.iterrows():
            summary = row[topic]["summary"]
            citations = row[topic]["citations"]
            if summary == "" or len(citations) == 0:
                continue
            row = {
                "topic": topic,
                "summary": summary,
                "citations": citations,
            }
            to_sql_table.append(row)
    return to_sql_table


# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument(
        "--txtfile",
        type=str,
        default="../data/davebuster/DAVEBUSTER'SENTERTAINMENTINC_20220629_8-K_EX-101_CreditLoanAgreement.PDF.txt",
    )
    parser.add_argument("--prompt_1", type=str, default="credit.pass1.prompt.txt")
    parser.add_argument("--prompt_2", type=str, default="credit.pass2.prompt.txt")
    parser.add_argument("--pass1results", type=str, default="output.json")
    args = parser.parse_args()

    if args.step == 1:
        print("Step 1")
        data = process_file_and_prompt(args.txtfile, args.prompt_1)
        print(f"Processed {len(data)} pages")
        # Write data to json file
        with open(args.pass1results, "w") as file:
            json.dump(data, file)
        print(f"Written pass 1 results to {args.pass1results}")

    elif args.step == 2:
        print("Step 2")
        df = read_and_clean_data(args.pass1results)
        print(f"Read {len(df)} rows from {args.pass1results}")
        topic_columns = df.columns.tolist()
        print(f"Topic columns: {topic_columns}")
        to_sql_table = filter_data_for_sql(df)
        # Write to a file
        pd.DataFrame(to_sql_table).to_csv("pass2.csv", index=False)
