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
import weaviate
from typing import List, Dict


class ChatGPTResponse(BaseModel):
    original_page_text: str
    page_number: int
    summary: str
    tranche: Optional[str] = None
    quantum: Optional[str] = None
    financial_maintenance_covenant: Optional[str] = None
    addbacks_cap: Optional[str] = None
    MFN_threshold: Optional[str] = None
    MFN_exceptions: Optional[str] = None
    portability: Optional[str] = None
    lender_counsel: Optional[str] = None
    borrower_counsel: Optional[str] = None
    borrower: Optional[str] = None
    guarantor: Optional[str] = None
    admin_agent: Optional[str] = None
    collat_agent: Optional[str] = None
    effective_date: Optional[str] = None


def get_weaviate_client() -> weaviate.Client:
    weaviate_credentials = json.loads(os.getenv("WEAVIATE_CREDENTIALS", "{}"))
    weaviate_url = weaviate_credentials.get("URL", "localhost:7000")
    weaviate_api_key = weaviate_credentials.get("API_KEY", "")
    weaviate_http_scheme = weaviate_credentials.get("HTTP_SCHEME", "http")

    auth_credentials = weaviate.AuthApiKey(api_key=weaviate_api_key)

    return weaviate.Client(
        url=f"{weaviate_http_scheme}://{weaviate_url}",
        auth_client_secret=auth_credentials,
    )
    # weaviate_client = weaviate.Client(
    #     embedded_options=weaviate.embedded.EmbeddedOptions(
    #         persistence_data_path="./weaviatedb",
    #     )
    # )
    # return weaviate_client


# Load the OPENAI_API_KEY key from an environment file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No OpenAI API key found in environment variables")
openai_client = OpenAI(api_key=api_key)


weaviate_client = get_weaviate_client()


# Load environment variables
load_dotenv()
# This Python script is designed to read a text file and a prompt file, process the text file by splitting it by page numbers, and then send each page along with the prompt to a ChatGPT API. The script handles these operations in parallel, using up to 8 workers for efficiency.


# Function to get text embedding
def get_text_embedding(text: str) -> List[float]:
    return (
        openai.embeddings.create(input=[text], model="text-embedding-ada-002")
        .data[0]
        .embedding
    )


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
def prepare_prompt(page_text: str, prompt_path: str) -> str:
    with open(prompt_path, "r") as file:
        prompt = file.read()
    TO_REPLACE = "{{.dataframe}}"
    # raise error if the prompt does not contain the placeholder
    if TO_REPLACE not in prompt:
        raise ValueError(f"Prompt file {prompt_path} does not contain {TO_REPLACE}")
    return prompt.replace(TO_REPLACE, page_text)


def send_to_chatgpt(input: str) -> ChatGPTResponse:
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"content": input, "role": "user"}],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if content is None:
        print(f"No content error processing page: {input}")
        raise Exception("No content error processing page")
    j = json.loads(content)
    return ChatGPTResponse(**j, original_page_text=input)


def handle_response(chatgpt_response: ChatGPTResponse):
    store_response_weaviate(chatgpt_response)
    print(chatgpt_response)


# Function to store the response in Weaviate
def store_response_weaviate(chatgpt_response: ChatGPTResponse):
    embedding = get_text_embedding(chatgpt_response.original_page_text)
    data_object = {
        "embedding": embedding,
        "original_page_text": chatgpt_response.original_page_text,
        "page_number": chatgpt_response.page_number,
        "summary": chatgpt_response.summary,
        "tranche": chatgpt_response.tranche,
        "quantum": chatgpt_response.quantum,
        "financial_maintenance_covenant": chatgpt_response.financial_maintenance_covenant,
        "addbacks_cap": chatgpt_response.addbacks_cap,
        "MFN_threshold": chatgpt_response.MFN_threshold,
        "MFN_exceptions": chatgpt_response.MFN_exceptions,
        "portability": chatgpt_response.portability,
        "lender_counsel": chatgpt_response.lender_counsel,
        "borrower_counsel": chatgpt_response.borrower_counsel,
        "borrower": chatgpt_response.borrower,
        "guarantor": chatgpt_response.guarantor,
        "admin_agent": chatgpt_response.admin_agent,
        "collat_agent": chatgpt_response.collat_agent,
        "effective_date": chatgpt_response.effective_date,
    }
    try:
        weaviate_client.data_object.create(data_object, "ChatGPTResponse")
        print("Stored in Weaviate:", chatgpt_response)
    except Exception as e:
        print("Error storing in Weaviate:", e)


def create_weaviate_schema():
    class_schema = {
        "class": "ChatGPTResponse",
        "description": "Response data from ChatGPT",
        "properties": [
            {"name": "original_page_text", "dataType": ["text"]},
            {"name": "page_number", "dataType": ["int"]},
            {"name": "summary", "dataType": ["text"]},
            {"name": "tranche", "dataType": ["text"]},
            {"name": "quantum", "dataType": ["text"]},
            {"name": "financial_maintenance_covenant", "dataType": ["text"]},
            {"name": "addbacks_cap", "dataType": ["text"]},
            {"name": "MFN_threshold", "dataType": ["text"]},
            {"name": "MFN_exceptions", "dataType": ["text"]},
            {"name": "portability", "dataType": ["text"]},
            {"name": "lender_counsel", "dataType": ["text"]},
            {"name": "borrower_counsel", "dataType": ["text"]},
            {"name": "borrower", "dataType": ["text"]},
            {"name": "guarantor", "dataType": ["text"]},
            {"name": "admin_agent", "dataType": ["text"]},
            {"name": "collat_agent", "dataType": ["text"]},
            {"name": "effective_date", "dataType": ["text"]},
        ],
    }

    try:
        weaviate_client.schema.create_class(class_schema)
        print("Schema created successfully")
    except Exception as e:
        print(f"Error creating schema in Weaviate: {e}")


# Main function to process the file and prompt
def process_file_and_prompt(txt_file: str, prompt_template: str):
    t0 = time.time()

    pages = read_and_split_file(txt_file)
    print(
        f"Read {len(pages)} pages from {txt_file} in {(time.time() - t0):.2f} seconds"
    )

    # Note PAGE_LIMIT is for testing purposes, so we don't have to wait for all pages to process before checking the results
    PAGE_LIMIT = 4
    for page in pages[:PAGE_LIMIT]:
        if page == "":
            continue
        input = prepare_prompt(page, prompt_template)
        response = send_to_chatgpt(input)
        handle_response(response)

    # 106 pages in 296 seconds (2.8 seconds per page)
    # t1 = time.time()
    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     futures = []
    #     for page in pages:
    #         if page == "":
    #             continue
    #         future = executor.submit(send_to_chatgpt, page, prompt_file)
    #         futures.append(future)
    #     for future in concurrent.futures.as_completed(futures):
    #         response = future.result()
    #         handle_response(response)

    print(f"Processed {len(pages)} pages in {time.time() - t1} seconds")


COLUMNS = """page_number
tranche
quantum
financial_maintenance_covenant
addbacks_cap
mFN_threshold
mFN_exceptions
portability
lender_counsel
borrower_counsel
borrower
guarantor
admin_agent
collat_agent
effective_date
summary
original_page_text
embedding"""


def read_from_weaviate() -> List[Dict]:
    query = f"""{{
        Get {{
            ChatGPTResponse {{
                {COLUMNS} 
            }}
        }}
    }}"""
    response = weaviate_client.query.raw(query)
    return response["data"]["Get"]["ChatGPTResponse"]


# Function to export data to CSV
def export_to_csv(data: List[Dict], file_path: str):
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=COLUMNS.split("\n"), quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(data)


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
    parser.add_argument("--prompt_2", type=str, default="credit.prompt.pass-2.new.txt")
    parser.add_argument("--csv", type=str, default="DAVEBUSTER.csv")
    args = parser.parse_args()

    if args.step == 1:
        print("Step 1")
        weaviate_client.schema.delete_class("ChatGPTResponse")
        create_weaviate_schema()
        process_file_and_prompt(args.txtfile, args.prompt_1)
        data = read_from_weaviate()
        export_to_csv(data, args.csv)

    # elif args.step == 2:
    #     print("Step 2")
    #     questions = pd.read_csv("questions.tsv", sep="|")
    #     questions["topics"] = questions["topics"].apply(lambda x: x.split(","))

    #     # data = read_from_weaviate()
    #     # df = pd.DataFrame(data)
    #     df = pd.read_csv(args.csv)
    #     print(f"Read {len(df)} rows from {args.csv}")
    #     dataframes_dict = {}
    #     # make topic columns without the page_number, summary, original_page_text, and embedding columns
    #     topic_columns = set(df.columns.tolist()) - set(
    #         ["page_number", "summary", "original_page_text", "embedding"]
    #     )
    #     print(f"Topic columns: {topic_columns}")

    #     # Create an Excel writer object
    #     xl_writer = pd.ExcelWriter("output.xlsx", engine="openpyxl")
    #     questions.to_excel(xl_writer, sheet_name="questions", index=False)
    #     for column in topic_columns:
    #         # Create a new DataFrame for the column where it is not null
    #         non_null_df = df[df[column].notnull()][
    #             ["page_number", "summary", column]
    #         ].sort_values(by=["page_number"])

    #         dataframes_dict[column] = non_null_df

    #         # Write each DataFrame to a different sheet
    #         non_null_df.to_excel(xl_writer, sheet_name=column, index=False)
    #         print(f"Written {len(non_null_df)} rows to {column} sheet")
    #     xl_writer.close()

    #     # You are a legal analyst. This is an data frame with the page_number, summary, and topic (e.g. addbacks_cap). I'm looking to answer the topic question for the summary. Based on the above information, what is the answer to the topic question? Synthesize the information from the topic column along with the page_number and summary for context.

    #     # # For each question, get the relevant data frame column and generate the prompt
    #     # read from args.prompt_2 file

    #     with open(args.prompt_2, "r") as f:
    #         prompt_2_template = f.read()

    #     for index, row in questions.iterrows():
    #         print(f"Processing question {index}: {row['question']}")
    #         question = row["question"]
    #         topics = row["topics"]
    #         if len(topics) == 0:
    #             continue
    #         elif len(topics) == 1:
    #             df = dataframes_dict[topics[0]]
    #         else:
    #             df = pd.concat([dataframes_dict[topic] for topic in topics])

    #         prompt = prompt_2(prompt_2_template, question, df)
    #         total_time = 0
    #         t0 = time.time()

    #         # Write the prompt to a file
    #         with open(f"Q{index}.part2.prompt.in.txt", "w") as file:
    #             file.write(prompt)
    #         # Send the prompt to ChatGPT
    #         response = part2_chatgpt(index, prompt)
    #         # Write the prompt to a file
    #         with open(f"Q{index}.part2.prompt.out.txt", "w") as file:
    #             file.write(response)

    #         t1 = time.time()
    #         total_time += t1 - t0
    #         print(
    #             f"Processed question {index} in {t1 - t0} seconds, total time: {total_time} seconds"
    #         )
