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
from pydantic import BaseModel
from typing import List, Dict
from sqlalchemy import create_engine, text
from more_itertools import flatten
import fitz  # PyMuPDF


MIN_CHARS = 120  # Minimum characters for a sentence
MAX_CHARS = 360  # Maximum characters for a sentence

load_dotenv()

url = os.getenv("POSTGRES_DB_CREDENTIALS")
engine = create_engine(url)  # type: ignore

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No OpenAI API key found in environment variables")
openai_client = OpenAI(api_key=api_key)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
# 3.5 turbo is cheap but 16k limit is pretty low
# MODEL_NAME = "gpt-4-1106-preview"
# MAX_TEXT_LENGTH = 4096 * 4  # 16KB for GPT-3.5 turbo

# Override
MODEL_NAME = "gpt-4-turbo-preview"
MAX_TEXT_LENGTH = 4096 * 8 * 4  # 128KB for GPT-4 turbo


def upload_file_new_doc_id(pdf_path: str) -> int:
    filename = os.path.basename(pdf_path)

    with engine.connect() as conn:
        transaction = conn.begin()  # Start a new transaction
        query = text("SELECT MAX(id) FROM experiments.documents;")
        result = conn.execute(query).fetchone()
        doc_id = result[0] if result[0] is not None else 0  # type: ignore
        new_doc_id = doc_id + 1

        # Read the PDF file as binary data
        with open(pdf_path, "rb") as file:
            pdf_data = file.read()

        # Insert the new document into the database with the raw PDF data
        insert_query = text(
            "INSERT INTO experiments.documents (id, filename, raw_file) VALUES (:id, :filename, :raw_file)"
        )
        conn.execute(
            insert_query,
            {"id": new_doc_id, "filename": filename, "raw_file": pdf_data},
        )
        transaction.commit()  # Explicitly commit the transaction
        print(f"Uploaded {filename} to document_id {new_doc_id}")

    return new_doc_id


class Box(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class TagParagraphBox(BaseModel):
    tag: str
    paragraph: str
    box: Box


def pdf2text_inject_tags(pdf_path) -> list[TagParagraphBox]:
    """
    Extracts bounding boxes and text of paragraphs from a PDF.

    :param pdf_path: Path to the PDF file.
    :return: A list of tuples, each containing the bounding box (as a rect) and the text of a paragraph.
    """
    segments: list[TagParagraphBox] = []

    with fitz.open(pdf_path) as doc:  # type: ignore
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("blocks")
            blocks.sort(
                key=lambda block: (block[1], block[0])
            )  # Sort blocks by y0, x0 (top to bottom, left to right)
            for block_idx, block in enumerate(blocks):
                rect = fitz.Rect(block[:4])  # The bounding box of the block
                text = block[4]  # The text content of the block
                tag = f"<P{page_num}S{block_idx}/>"
                segments.append(
                    TagParagraphBox(
                        tag=tag,
                        paragraph=text,
                        box=Box(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1),
                    )
                )

    print(f"Text extracted from {pdf_path}, bounding boxes {len(segments)}.")
    return segments


def read_and_split_file(document_id: int) -> List[str]:
    """
    Read the file and split the text into pages of MAX_TEXT_LENGTH.
    A page is defined as a string of text with a maximum length of MAX_TEXT_LENGTH.
    It does not mean a page in the traditional book sense.
    """
    pages = []
    df_tag_paragraph = get_document_tags_paragraphs(document_id)
    current_page = ""
    for tag, paragraph in df_tag_paragraph.itertuples(index=False):
        if len(paragraph) + len(current_page) > MAX_TEXT_LENGTH / 8:
            pages.append(current_page)
            current_page = tag + " " + paragraph
        else:
            current_page += " " + (tag if tag else "") + " " + paragraph
    # Don't forget to add the last page if it's not empty
    if current_page:
        pages.append(current_page)
    return pages


# Function to read and prepare the prompt
def prepare_prompt(page_text: str, prompt_name: str) -> str:
    # with open(prompt_path, "r") as file:
    #     prompt = file.read()
    prompt = get_prompt_from_sql(prompt_name)
    TO_REPLACE = "{{.dataframe}}"
    # raise error if the prompt does not contain the placeholder
    if TO_REPLACE not in prompt:
        raise ValueError(f"Prompt file {prompt_name} does not contain {TO_REPLACE}")
    gpt_input = prompt.replace(TO_REPLACE, page_text)

    return gpt_input


def send_to_chatgpt(input: str, page_number: int) -> tuple[dict, float]:
    print(f"Sending to {MODEL_NAME} for page {page_number}...")
    if len(input) > MAX_TEXT_LENGTH:
        raise ValueError(
            f"Input length {len(input)} exceeds maximum length {MAX_TEXT_LENGTH}"
        )
    start = time.time()
    try:
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"content": input, "role": "user"}],
            response_format={"type": "json_object"},
        )
    except openai.RateLimitError as e:
        print(f"Rate limit error processing page {page_number}: {str(e)}")
        print("Waiting 30s before retrying...")
        time.sleep(30)
        return send_to_chatgpt(input, page_number)

    content = response.choices[0].message.content
    if content is None:
        print(f"No content error processing page: {input}")
        raise Exception(f"No content error processing page {page_number}")
    json_response = json.loads(content)
    end_time = time.time() - start
    print(f"Processed page {page_number} in {end_time} seconds")
    return (json_response, end_time)


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


class PromptIO(BaseModel):
    """
    A class to represent the input and output prompt files.
    """

    prompt_in: str
    prompt_out: str
    page: int
    time_taken_seconds: float
    model: str


# Main function to process the file and prompt
def process_step_1_file_and_prompt(
    document_id: int,
    prompt_template: str,
    limit: int | None,
    fast: bool = True,
) -> List[PromptIO]:
    t0 = time.time()
    promptios: list[PromptIO] = []
    pages = read_and_split_file(document_id)
    print(
        f"Read {len(pages)} pages from document_id {document_id} in {(time.time() - t0):.2f} seconds"
    )
    t1 = time.time()

    if not fast:
        for page_num, page_data in enumerate(pages[:limit]):
            print(f"Processing page {pages.index(page_data) + 1} of {len(pages)}")
            if page_data == "":
                continue
            input = prepare_prompt(page_data, prompt_template)

            response, seconds = send_to_chatgpt(input, page_num + 1)
            promptios.append(
                PromptIO(
                    prompt_in=input,
                    prompt_out=json.dumps(response, indent=2),
                    page=page_num,
                    time_taken_seconds=seconds,
                    model=MODEL_NAME,
                )
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for page_num, page_data in enumerate(pages[:limit]):
                print(f"Processing page {page_num + 1} of {len(pages)}")
                if page_data == "":
                    continue

                # Use a task function to encapsulate the call with context
                def task(p_num=page_num, p_data=page_data) -> PromptIO:
                    input = prepare_prompt(p_data, prompt_template)
                    response, seconds = send_to_chatgpt(input, p_num + 1)
                    return PromptIO(
                        prompt_in=input,
                        prompt_out=json.dumps(response, indent=2),
                        page=p_num,
                        time_taken_seconds=seconds,
                        model=MODEL_NAME,
                    )

                futures.append(executor.submit(task))

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                promptios.append(result)

    print(f"Processed {len(pages)} pages in {time.time() - t1} seconds")
    return promptios


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


# Function to parse the custom-formatted strings in 'citations' column
def parse_citations(citation_str):
    # Remove the curly braces and split the string into individual elements
    elements = citation_str.strip("{}").split(",")
    # Clean each element by removing unwanted characters
    # parsed_elements = [elem.strip("<>/") for elem in elements]
    return elements


def filter_tags_with_surroundings(df, tags, surrounding=0):
    """
    Filters a DataFrame to include rows where the tag column value is in the specified set of tags,
    along with three surrounding tags above and below each match.
    NOTE: Large of surrounding could result in a large dataframe,
    likely to be too large for the model to process.

    Parameters:
    - df: pandas DataFrame with a 'tag' column.
    - tags: Set of tag values to filter by.

    Returns:
    - A filtered pandas DataFrame.
    """
    # First, find the indices of rows where the tag column value is in the tags set
    matching_indices = df.index[df["tag"].isin(tags)].tolist()

    # Initialize a set to hold all indices to include (for deduplication)
    all_indices = set()

    # For each matching index, add it and the three surrounding indices above and below
    for idx in matching_indices:
        # Calculate ranges, ensuring we don't go out of bounds
        start_idx = max(idx - surrounding, 0)
        end_idx = min(
            idx + surrounding + 1, len(df)
        )  # +1 because range end is exclusive

        # Add the range of indices to the set
        all_indices.update(range(start_idx, end_idx))

    # Convert the set of indices back to a sorted list
    final_indices = sorted(list(all_indices))

    # Use the final indices to filter the DataFrame
    filtered_df = df.loc[final_indices, ["tag", "paragraph"]]

    return filtered_df


def create_collect_prompts(template_path: str, responses: pd.DataFrame) -> List[str]:
    """
    Prompt 2 needs to look up the citations column and for each citation string,
    get the relevant quote from the summary column in documents_tags table
    and then ask the question to the model.
    """
    # Read the template_path file
    template = read_prompt_from_sql(template_path)
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
    document_id = responses["document_id"].iloc[0]
    query = f"SELECT tag, paragraph FROM experiments.documents_tags WHERE document_id = '{document_id}'"
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
            tags = sorted(set(flatten(df_citations.to_list())))
        else:
            all_citations = [
                citation
                for sublist in filtered_df["citations"].apply(parse_citations)
                for citation in sublist
            ]
            unique_citations = set(all_citations)
            df_citations = unique_citations
            tags = sorted(df_citations)

        filtered_df_documents_tags = filter_tags_with_surroundings(
            df_documents_tags, tags
        )

        combined_strings = filtered_df_documents_tags.apply(
            lambda x: f"{x['tag']} {x['paragraph']}", axis=1
        ).tolist()
        combined_citations_string = "\n".join(combined_strings)
        topic_prompt = topic_prompt.replace(
            CITATIONS_PLACEHOLDER, combined_citations_string
        )

        prompts.append(topic_prompt)

    return prompts


def send_chatgpt_2(index: int, prompt: str) -> tuple[str, float]:
    try:
        print(f"Sending question {index}...")
        if len(prompt) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Input length {len(prompt)} exceeds maximum length {MAX_TEXT_LENGTH}"
            )
        t0 = time.time()
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"content": prompt, "role": "user"}],
        )
        content = response.choices[0].message.content
        seconds = time.time() - t0
        print(f"Processed question {index} in {seconds:.2f} seconds")
        if content is None:
            print(f"No content error processing question {index}: {response}")
            return "", seconds
        return content, seconds

    except openai.OpenAIError as e:
        # Handle any errors from the OpenAI API
        return f"Error processing question {index}: {str(e)}", 0.0


def read_and_clean_data(file_path):
    """Read JSON data from a dumped json file."""
    with open(file_path, "r") as file:
        data = json.load(file)
        data = pd.DataFrame(data)["prompt_out"].apply(json.loads)
        data = pd.DataFrame(data.to_list())

    return data


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
            if not summary or summary == "" or not citations or len(citations) == 0:
                continue
            row = {
                "topic": topic,
                "summary": summary,
                "citations": citations,
            }
            to_sql_table.append(row)
    return pd.DataFrame(to_sql_table)


def extract_json_data(promptio: PromptIO) -> dict:
    file_content = promptio.prompt_out
    # Check and remove first and last line as they contain ```json and ```, respectively
    if file_content.splitlines()[0] == "```json":
        joined = "".join(file_content.splitlines()[1:-1])
        data = json.loads(joined)
    else:
        data = json.loads(file_content)
    return fix_malformed_json(data)


def process_step_2(prompt_2, to_sql_table) -> List[PromptIO]:
    prompts = create_collect_prompts(
        prompt_2,
        to_sql_table,
    )
    print(f"Created {len(prompts)} topic prompts for pass 2.")
    promptios: List[PromptIO] = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for prompt_num, prompt_input in enumerate(prompts):
            if prompt_input == "":
                continue

            # Use a task function to encapsulate the call with context
            def task(p_num=prompt_num, p_data=prompt_input) -> PromptIO:
                response, seconds = send_chatgpt_2(p_num + 1, p_data)
                return PromptIO(
                    prompt_in=prompt_input,
                    prompt_out=response,
                    page=p_num,
                    time_taken_seconds=seconds,
                    model=MODEL_NAME,
                )

            futures.append(executor.submit(task))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            promptios.append(result)

    print(f"Processed {len(promptios)} pages in {(time.time() - t0):.2f} seconds")
    return promptios


def promptios_to_df(promptios: List[PromptIO]) -> pd.DataFrame:
    """
    Convert the list of PromptIO objects to a pandas DataFrame.
    Columns: prompt_in, prompt_out, page, time_taken_seconds, model
    """
    return pd.DataFrame([obj.model_dump() for obj in promptios])


def promptios_to_sql(promptios: List[PromptIO], document_id: int, step: int):
    """
    Write the list of PromptIO objects to the database table experiments.promptios
    For metrics and logging purposes, we also calculate the length of the input and output prompts.
    """
    pdf = promptios_to_df(promptios)
    pdf["document_id"] = document_id
    pdf["step"] = step
    pdf["length_out"] = pdf["prompt_out"].apply(len)
    pdf["length_in"] = pdf["prompt_in"].apply(len)
    pdf.to_sql("promptios", engine, if_exists="append", schema="experiments")


def get_document_tags_paragraphs(document_id: int) -> pd.DataFrame:
    query = f"SELECT tag, paragraph FROM experiments.documents_tags WHERE document_id = '{document_id}'"
    return pd.read_sql_query(query, engine)


def get_prompt_from_sql(prompt_name: str) -> str:
    query = (
        f"SELECT content FROM experiments.prompts WHERE prompt_name = '{prompt_name}'"
    )
    with engine.connect() as conn:
        result = conn.execute(text(query))
        prompt = result.scalar()
        if prompt is None:
            raise ValueError(f"No prompt found for {prompt_name}")
        return prompt


def write_prompt_file_to_sql(prompt_path: str):
    prompt_name = os.path.basename(prompt_path)
    with open(
        prompt_path, "r"
    ) as file:  # Fixed path issue and ensure file is properly closed
        content = file.read()

    # SQL command for upsert (insert or update on conflict)
    upsert_sql = text(
        """
        INSERT INTO experiments.prompts (prompt_name, content, updated_at) 
        VALUES (:prompt_name, :content, NOW()) 
        ON CONFLICT (prompt_name) DO UPDATE 
        SET content = EXCLUDED.content;
    """
    )

    # Execute the upsert command
    with engine.connect() as conn:
        conn.execute(upsert_sql, {"prompt_name": prompt_name, "content": content})
        conn.commit()

    # Check it can be read
    get_prompt_from_sql(prompt_name)


def read_pass1_results_from_sql(document_id: int) -> pd.DataFrame:
    query = (
        f"SELECT * FROM experiments.pass1_results WHERE document_id = '{document_id}'"
    )
    return pd.read_sql_query(query, engine)


def fix_malformed_json(json_obj) -> dict:
    """
    Detects and fixes malformed JSON structure for specific keys.

    Parameters:
    - json_obj: A dictionary representing the JSON object.

    Returns:
    - A dictionary with the corrected JSON structure.
    """
    # Define the expected correct structure
    correct_keys = ["field_name", "value", "citations", "explanation"]

    # Check if json_obj is missing any of the correct keys and has unexpected keys
    if (
        not all(key in json_obj for key in correct_keys)
        and "deal_structure" in json_obj
    ):
        # Fix the structure
        fixed_json = {
            "field_name": "deal_structure",
            "value": json_obj.pop(
                "deal_structure"
            ),  # Move the value and remove the key
            "citations": json_obj.get("citations", []),
            "explanation": json_obj.get("explanation", ""),
        }
        return fixed_json
    # If json_obj already has the correct structure or doesn't need fixing
    return json_obj


# # Example usage
# malformed_json = {
#     'deal_structure': 'Asset and Stock Sale/Purchase',
#     'citations': ['<P5S6/>', '<P30S34/>', '<P37S4/>', '<P37S5/>', '<P37S6/>', '<P68S1/>', '<P110S3/>', '<P110S4/>', '<P128S0/>'],
#     'explanation': "The transaction contains elements of both an asset sale/purchase and a stock sale/purchase. Citations indicate the sale and purchase of equity interests (which aligns with a stock sale/purchase), such as where sellers desire to sell and purchasers desire to acquire transferred equity interests. Other citations mention the structure as an asset sale/purchase, with references to treating the sale for tax purposes as an asset purchase and a step-up in tax basis for all assets involved. The combination of these factors and specific references to the transaction's structure lead to the conclusion that this is an Asset and Stock Sale/Purchase."
# }
# fixed_json = fix_malformed_json(malformed_json)
# print(json.dumps(fixed_json, indent=2))


def tagged_text_process(
    document_id: int,
    prompt_name_1: str,
    prompt_name_2: str,
    limit: int | None = None,
):
    """
    Process the file and prompts to generate the pass 2 prompts and responses.

    """
    print("Step 1")
    promptios_1 = process_step_1_file_and_prompt(
        document_id, prompt_name_1, limit=limit, fast=True
    )
    promptios_to_sql(promptios_1, document_id, 1)

    df_pass2_src = pd.DataFrame(map(lambda p: json.loads(p.prompt_out), promptios_1))
    topic_columns = df_pass2_src.columns.tolist()
    print(f"Topic columns: {topic_columns}")
    to_sql_table = filter_data_for_sql(df_pass2_src)

    print("Step 2")
    # Write to the database table experiments.pass1_results
    to_sql_table["document_id"] = document_id
    to_sql_table["model"] = MODEL_NAME
    to_sql_table.to_sql(
        "pass1_results", engine, if_exists="append", schema="experiments"
    )
    print(f"Written {len(to_sql_table)} records to experiments.pass1_results")

    # to_sql_table = read_pass1_results_from_sql(document_id)

    promptios_2 = process_step_2(prompt_name_2, to_sql_table)
    promptios_to_sql(promptios_2, document_id, 2)

    responses = list(map(extract_json_data, promptios_2))
    df_pass2_output = pd.DataFrame(responses)
    df_pass2_output["document_id"] = document_id
    df_pass2_output["model"] = MODEL_NAME
    df_pass2_output.to_sql(
        "pass2_results", engine, if_exists="append", schema="experiments"
    )
    print(f"Written {len(df_pass2_output)} records to experiments.pass2_results")
    df_pass2_output.to_csv("pass2.result.csv")
    print("Written pass 2 results to pass2.result.csv")


def write_document_tags_to_sql(boxes: list[TagParagraphBox], document_id: int) -> int:
    print(f"Writing document tags to SQL for document_id {document_id}")

    # Convert TagParagraphBox instances to a list of dicts with the box converted to an array
    records = [
        {
            "tag": box.tag,
            "paragraph": box.paragraph,
            "bounding_box": [box.box.x0, box.box.y0, box.box.x1, box.box.y1],
            "document_id": document_id,
        }
        for box in boxes
    ]

    # Convert records to a DataFrame
    df = pd.DataFrame(records)

    # Assuming 'engine' is already defined and connected to your PostgreSQL database
    df.to_sql(
        "documents_tags", engine, index=False, if_exists="append", schema="experiments"
    )
    return int(document_id)


def read_prompt_from_sql(prompt_name: str) -> str:
    query = (
        f"SELECT content FROM experiments.prompts WHERE prompt_name = '{prompt_name}'"
    )
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return result.scalar()  # type: ignore


def do_qa(pdf_path: str, prompt_1: str, prompt_2: str, limit: int | None = None):
    document_text_with_tags = pdf2text_inject_tags(pdf_path)
    document_id = upload_file_new_doc_id(pdf_path)
    print(f"document_id: {document_id}")
    write_document_tags_to_sql(document_text_with_tags, document_id)
    tagged_text_process(document_id, prompt_1, prompt_2, limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        type=str,
        default="../data/m&a/Arco_Platform_Ltd_Investment_Group_477m_Announce_20221130_merger_agree_20230811.pdf",
        # default="../data/m&a/Lumen_Incumbent_Local_Exchange_Carrier_Business_Apollo_Global_Management_LLC_7_500m_Announce_20210803_merger_agree_20210804.pdf",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="../data/m&a",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    prompt_1 = "ma.pass1.prompt.txt"
    prompt_2 = "ma.pass2.prompt.txt"
    write_prompt_file_to_sql(prompt_1)
    write_prompt_file_to_sql(prompt_2)

    # do_qa(args.pdf, prompt_1, prompt_2, args.limit)

    for file in os.listdir(args.dir):
        if file.endswith(".pdf"):
            print(file)
            do_qa(os.path.join(args.dir, file), prompt_1, prompt_2, args.limit)
            print(f"Processed {file}")
            time.sleep(3)
