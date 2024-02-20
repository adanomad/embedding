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
from sqlalchemy import create_engine, text
import ast
from more_itertools import flatten

# MODEL_NAME = "gpt-4-1106-preview"
MODEL_NAME = "gpt-3.5-turbo"
MAX_TEXT_LENGTH = 4096 * 4  # 16KB

load_dotenv()

url = os.getenv("POSTGRES_DB_CREDENTIALS")
engine = create_engine(url)  # type: ignore

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
    print(f"Sending to {MODEL_NAME} for page {page_number}...")
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


def extract_string_between_tags(body: str, tag: str) -> str | None:
    """
    Extracts and returns the string content between the provided XML-style tag.

    Parameters:
    - body (str): The body of text from which to extract the content.
    - tag (str): The tag name whose content is to be extracted.

    Returns:
    - str: The extracted string content between the opening and closing tags. If the tag is not found, returns None.
    """
    # Create a regular expression pattern to match the content between the specified tags
    pattern = f"<{tag}>(.*?)</{tag}>"

    # Search for the pattern in the body
    match = re.search(pattern, body, re.DOTALL)

    # If a match is found, return the first group (content between the tags); otherwise, return None
    return match.group(1) if match else None


def find_line_with_needle(body: str, needle: str) -> str | None:
    """
    Searches for a line in the given body that contains the specified needle.

    Parameters:
    - body (str): The string body to search through, where lines are separated by newlines.
    - needle (str): The string to search for within the lines of the body.

    Returns:
    - str: The first line that contains the needle, or None if no such line is found.
    """
    # Split the body into lines
    lines = body.split("\n")

    # Iterate through each line and check if it contains the needle
    for line in lines:
        if needle in line:
            return line  # Return the first line that contains the needle

    return None  # Return None if the needle is not found in any line


def replace_text_inside_tag(body: str, tag: str, replacement_str: str) -> str:
    """
    Replaces the body found between opening and closing XML-style tags with a provided replacement string.

    Parameters:
    - body (str): The original body containing the XML-style tags.
    - tag (str): The tag name whose content is to be replaced.
    - replacement_str (str): The string to insert between the opening and closing tags.

    Returns:
    - str: The modified body with the replacement string inserted between the specified tags.
    """
    # Define the pattern to match body between the specified tags (including multiline content)
    pattern = f"<{tag}>(.*?)</{tag}>"

    # Use re.sub() to replace the content between the tags with the replacement string
    # The replacement also includes the opening and closing tags themselves
    replaced_text = re.sub(
        pattern, f"<{tag}>{replacement_str}</{tag}>", body, flags=re.DOTALL
    )

    return replaced_text


def create_collect_prompts(template_path: str, responses: pd.DataFrame) -> List[str]:
    """
    Prompt 2 needs to look up the citations column and for each citation string,
    get the relevant quote from the summary column in documents_tags table
    and then ask the question to the model.
    """
    # The response may have whitespace, so we need to strip it
    responses = responses.map(lambda x: x.strip() if isinstance(x, str) else x)
    responses.columns = responses.columns.str.strip()

    # Read the template_path file
    with open(template_path, "r") as file:
        template = file.read()
    DATAFRAME_PLACEHOLDER = "{{.dataframe}}"
    CITATIONS_PLACEHOLDER = "{{.citations}}"
    if DATAFRAME_PLACEHOLDER not in template:
        raise ValueError(
            f"Prompt file {template} does not contain {DATAFRAME_PLACEHOLDER}"
        )
    if CITATIONS_PLACEHOLDER not in template:
        raise ValueError(
            f"Prompt file {template} does not contain {CITATIONS_PLACEHOLDER}"
        )
    terminologies = extract_string_between_tags(template, "terminology")
    if terminologies is None:
        raise ValueError("No terminology found in the template")

    # Read documents_tags table
    query = "SELECT * FROM experiments.documents_tags WHERE document_id = 2"
    df_documents_tags = pd.read_sql_query(query, engine)
    unique_topics = responses["topic"].unique()
    prompts = []
    for topic in unique_topics:
        topic_definition = find_line_with_needle(terminologies, topic)
        if topic_definition is None:
            raise ValueError(f"No definition found for {topic} in the template")
        topic_prompt = replace_text_inside_tag(
            template, "terminology", topic_definition
        )
        filtered_df = responses[responses["topic"] == topic]
        dataframe_texts = (
            filtered_df[["citations", "summary"]]
            .apply(lambda x: f"{x['citations']} {x['summary']}", axis=1)
            .to_list()
        )
        topic_prompt = topic_prompt.replace(
            DATAFRAME_PLACEHOLDER, "\n".join(dataframe_texts)
        )
        print(f"Topic {topic}: {len(filtered_df)} rows")
        # Get relevant tags from documents_tags
        # Check if it's a list of strings or a string
        if isinstance(filtered_df["citations"].iloc[0], list):
            df_citations = filtered_df["citations"]
        else:
            df_citations = filtered_df["citations"].apply(ast.literal_eval)

        tags = sorted(set(flatten(df_citations.to_list())))
        # filter df_documents_tags to only include rows where the tag column value is in the tags set
        filtered_df_documents_tags = df_documents_tags[
            df_documents_tags["tag"].isin(tags)
        ][["tag", "line"]]

        combined_strings = filtered_df_documents_tags.apply(
            lambda x: f"{x['tag']} {x['line']}", axis=1
        ).tolist()
        combined_string = "\n".join(combined_strings)
        topic_prompt = topic_prompt.replace(CITATIONS_PLACEHOLDER, combined_string)

        prompts.append(topic_prompt)

    return prompts


def part2_chatgpt(index: int, prompt: str) -> str:
    try:
        print(f"Sending question {index}...")
        response = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"content": prompt, "role": "user"}],
        )
        content = response.choices[0].message.content
        print(f"Processed question {index}")
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


def filter_data_for_sql(df) -> pd.DataFrame:
    """Filter data to prepare rows for SQL insertion."""
    to_sql_table = []
    for topic in df.columns:
        filtered_df = df[df[topic].notna()][[topic]]
        print(f"Transformed {len(filtered_df)} rows for {topic}")
        for _, row in filtered_df.iterrows():
            if not row[topic]:
                continue
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
    return pd.DataFrame(to_sql_table)


def process(
    txtfile: str,
    prompt_1: str,
    prompt_2: str,
):
    """
    Process the file and prompts to generate the pass 2 prompts and responses.

    """
    # print("Step 1")
    PASS1RESULTS = "pass1.results.json"

    data = process_file_and_prompt(txtfile, prompt_1)
    print(f"Processed {len(data)} pages")
    # Write data to json file
    with open(PASS1RESULTS, "w") as file:
        json.dump(data, file)
    print(f"Written pass 1 results to {PASS1RESULTS}")

    # # Read it back
    print("Step 2")
    df = read_and_clean_data(PASS1RESULTS)
    print(f"Read {len(df)} rows from {PASS1RESULTS}")
    topic_columns = df.columns.tolist()
    print(f"Topic columns: {topic_columns}")
    to_sql_table = filter_data_for_sql(df)
    # # Write to a file
    pd.DataFrame(to_sql_table).to_csv("pass2.csv", index=False)
    prompts = create_collect_prompts(
        prompt_2,
        to_sql_table,
    )
    print(f"Created {len(prompts)} prompts")
    # Write prompts to file
    for idx, prompt in enumerate(prompts):
        with open(f"pass2.{idx + 1}.prompt.in.txt", "w") as file:
            file.write(prompt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for idx, prompt in enumerate(prompts):
            futures.append(executor.submit(part2_chatgpt, idx, prompt))
        responses = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
    # Write responses to file
    for idx, response in enumerate(responses):
        with open(f"pass2.{idx + 1}.prompt.out.txt", "w") as file:
            file.write(response)
    print(f"Written {len(responses)} responses to pass2.*.prompt.out.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--txtfile",
        type=str,
        default="../data/m&a/arco/Arco_Platform_Ltd_Investment_Group_477m_Announce_20221130_merger_agree_20230811.pdf.txt",
    )
    parser.add_argument("--prompt_1", type=str, default="credit.pass1.prompt.txt")
    parser.add_argument("--prompt_2", type=str, default="credit.pass2.prompt.txt")
    args = parser.parse_args()

    process(args.txtfile, args.prompt_1, args.prompt_2)
