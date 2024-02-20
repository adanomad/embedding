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
import ast
from more_itertools import flatten
import fitz  # PyMuPDF
from df_psql import get_new_doc_id

# MODEL_NAME = "gpt-4-1106-preview"
MODEL_NAME = "gpt-3.5-turbo"
MAX_TEXT_LENGTH = 4096 * 4  # 16KB
MIN_CHARS = 120  # Minimum characters for a sentence
MAX_CHARS = 360  # Maximum characters for a sentence

load_dotenv()

url = os.getenv("POSTGRES_DB_CREDENTIALS")
engine = create_engine(url)  # type: ignore

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No OpenAI API key found in environment variables")
openai_client = OpenAI(api_key=api_key)


def pdf2text_inject_tags(pdf_path, inject_tokens=True) -> str:

    # Open the PDF file
    all_text_file = ""
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            text = page.get_text()
            if inject_tokens:
                # This could cause string find issues if the text is not found in original document
                text = text.replace("\n", " ")
                text = re.sub(r"\s+", " ", text).strip()
                indexed_sentences = extract_segments(text)
                text = inject_indices(indexed_sentences, page_num)
                all_text_file += text
            else:
                all_text_file += f"<PAGE {page_num}/>\n"

    print(f"Text and images extracted from {pdf_path}")
    return all_text_file


def extract_segments(text: str) -> list[str]:
    # Define patterns for identifying key segments; these can be adjusted or expanded based on the document's structure
    patterns = [
        # Match Roman numerals in parentheses. This pattern is simplified and might not cover all edge cases.
        r"\((?=[MDCLXVI])(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))\)",
        # Alphabetical items in parentheses (unchanged, as it suits the requirement)
        r"\([a-z]\)",
        # Adding a pattern for semicolons as a potential separator for items within the same sentence
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
    if not segment_indices:
        return [text]
    for i in range(len(segment_indices) - 1):
        # Extract text segments based on the indices
        segments.append(text[segment_indices[i] : segment_indices[i + 1]])

    # Add the final segment if not captured
    if segment_indices:
        segments.append(text[segment_indices[-1] :])

    return segments


def inject_indices_simple(indexed_sentences: list[str], page: int) -> str:
    # Initialize an empty string to hold the result
    combined_with_indices = ""

    # Loop through the list of indexed sentences
    for i, sentence in enumerate(indexed_sentences, start=1):
        # Append the sentence with its preceding index
        # Assuming <i> is the tag before the sentence and <i+1> is the tag after the sentence
        # For the last sentence, there's no next sentence, so just use <i>
        combined_with_indices += f"<PAGE{page}SEGMENT{i}/>{sentence}\n"

    return combined_with_indices


def normalize_sentence_lengths(
    sentences: list[str], min_chars: int, max_chars: int
) -> list[str]:
    """
    Adjust sentences to ensure each is within the specified minimum and maximum character lengths.
    Sentences shorter than min_chars are merged with subsequent sentences, and sentences longer
    than max_chars are split at suitable points.
    """

    processed_sentences = []
    i = 0

    # First pass: merge short sentences
    while i < len(sentences):
        current_sentence = sentences[i]
        if len(current_sentence) < min_chars and i + 1 < len(sentences):
            # Ensure not to exceed list bounds
            next_sentence = sentences[i + 1]
            merged_sentence = current_sentence + " " + next_sentence
            # Check if the merged sentence still needs to be split (if it exceeds max_chars)
            if len(merged_sentence) <= max_chars or max_chars == -1:
                processed_sentences.append(merged_sentence)
                i += 2  # Skip the next sentence as it's merged
            else:
                # If merging results in a sentence longer than max_chars, don't merge
                processed_sentences.append(current_sentence)
                i += 1  # Only increment by 1 to re-evaluate next_sentence in the next loop
        else:
            processed_sentences.append(current_sentence)
            i += 1

    # Second pass: split long sentences
    final_sentences = []
    for sentence in processed_sentences:
        if len(sentence) > max_chars and max_chars != -1:
            split_index = max(
                sentence.rfind(",", 0, max_chars), sentence.rfind(".", 0, max_chars)
            )
            if split_index != -1:
                # Split the sentence at the last comma/period before max_chars
                first_part = sentence[: split_index + 1]
                second_part = sentence[split_index + 2 :].strip()
                final_sentences.extend([first_part, second_part])
            else:
                # No suitable split point, append the sentence as is
                final_sentences.append(sentence)
        else:
            # Sentence is within the acceptable length range
            final_sentences.append(sentence)

    return final_sentences


def inject_indices(sentences: list[str], page: int) -> str:
    """Inject indices into sentences, merging short ones and splitting long ones."""
    final_sentences = normalize_sentence_lengths(sentences, MIN_CHARS, MAX_CHARS)
    # final_sentences = sentences

    combined_with_indices = ""
    for index, sentence in enumerate(final_sentences, start=1):
        combined_with_indices += f"<P{page}S{index}/>{sentence}\n"

    return combined_with_indices.strip()


def read_and_split_file(document_id: str) -> List[str]:
    """
    Read the file and split the text into pages of MAX_TEXT_LENGTH.
    A page is defined as a string of text with a maximum length of MAX_TEXT_LENGTH.
    It does not mean a page in the traditional book sense.
    """
    pages = []
    df_tag_paragraph = get_document_tags(document_id)
    current_page = ""
    for tag, paragraph in df_tag_paragraph.itertuples(index=False):
        if len(paragraph) + len(current_page) > MAX_TEXT_LENGTH:
            pages.append(current_page)
            current_page = tag + " " + paragraph
        else:
            current_page += " " + (tag if tag else "" + " " + paragraph)
    # Don't forget to add the last page if it's not empty
    if current_page:
        pages.append(current_page)
    return pages


# Function to read and prepare the prompt
def prepare_prompt(page_text: str, prompt_name: str, page_number: int) -> str:
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
    document_id: str,
    prompt_template: str,
    limit: int | None,
    fast: bool = True,
) -> List[PromptIO]:
    t0 = time.time()
    promptios: list[PromptIO] = []
    pages = read_and_split_file(document_id)
    print(
        f"Read {len(pages)} pages from {document_id} in {(time.time() - t0):.2f} seconds"
    )
    t1 = time.time()

    if not fast:
        for page_num, page_data in enumerate(pages[:limit]):
            print(f"Processing page {pages.index(page_data) + 1} of {len(pages)}")
            if page_data == "":
                continue
            input = prepare_prompt(page_data, prompt_template, page_num)

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
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for page_num, page_data in enumerate(pages[:limit]):
                print(f"Processing page {page_num + 1} of {len(pages)}")
                if page_data == "":
                    continue

                # Use a task function to encapsulate the call with context
                def task(p_num=page_num, p_data=page_data) -> PromptIO:
                    input = prepare_prompt(p_data, prompt_template, p_num)
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
        ][["tag", "paragraph"]]

        combined_strings = filtered_df_documents_tags.apply(
            lambda x: f"{x['tag']} {x['paragraph']}", axis=1
        ).tolist()
        combined_string = "\n".join(combined_strings)
        topic_prompt = topic_prompt.replace(CITATIONS_PLACEHOLDER, combined_string)

        prompts.append(topic_prompt)

    return prompts


def part2_chatgpt(index: int, prompt: str) -> tuple[str, float]:
    try:
        print(f"Sending question {index}...")
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
            if summary == "" or len(citations) == 0:
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

    return data


def process_step_2(prompt_2, to_sql_table) -> List[PromptIO]:
    prompts = create_collect_prompts(
        prompt_2,
        to_sql_table,
    )
    print(f"Created {len(prompts)} topic prompts for pass 2.")
    promptios: List[PromptIO] = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for prompt_num, prompt_input in enumerate(prompts):
            if prompt_input == "":
                continue

            # Use a task function to encapsulate the call with context
            def task(p_num=prompt_num, p_data=prompt_input) -> PromptIO:
                response, seconds = part2_chatgpt(p_num + 1, p_data)
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


def promptios_to_sql(promptios: List[PromptIO], document_id: str, step: int):
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


def get_document_tags(document_id: str) -> pd.DataFrame:
    query = f"SELECT tag, paragraph FROM experiments.documents_tags WHERE document_id = {document_id}"
    return pd.read_sql_query(query, engine)


def get_prompt_from_sql(prompt_name: str) -> str:
    query = (
        f"SELECT content FROM experiments.prompts WHERE prompt_name = '{prompt_name}'"
    )
    with engine.connect() as conn:
        return conn.execute(text(query)).scalar()


def write_prompt_file_to_sql(prompt_name: str):
    content = open(prompt_name, "r").read()
    df = pd.DataFrame([{"prompt_name": prompt_name, "content": content}])
    df.to_sql("prompts", engine, if_exists="append", schema="experiments", index=False)


def tagged_text_process(
    document_id: str,
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


def write_document_tags_to_sql(filename: str, document_content: str) -> str:
    document_id = get_new_doc_id(filename)
    print(f"Writing document tags to SQL for document_id {document_id}")
    # split document_content into sentences using regex looking for tag structure of <P\d+S\d+/>
    sentences = re.split(r"(<P\d+S\d+/>)", document_content)

    # Initialize an empty list to hold the structured data
    structured_data = []

    # Iterate through the sentences list two steps at a time
    for i in range(0, len(sentences), 2):
        content = sentences[i]
        tag = sentences[i + 1] if i + 1 < len(sentences) else None
        structured_data.append({"tag": tag, "paragraph": content})
    df = pd.DataFrame(structured_data)
    df["document_id"] = document_id
    df.to_sql(
        "documents_tags", engine, index=False, if_exists="append", schema="experiments"
    )
    return document_id


def read_prompt_from_sql(prompt_name: str) -> pd.DataFrame:
    query = f"SELECT * FROM experiments.promptios WHERE prompt_name = {prompt_name}"
    return pd.read_sql_query(query, engine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--txtfile",
        type=str,
        # default="../data/m&a/arco/Arco_Platform_Ltd_Investment_Group_477m_Announce_20221130_merger_agree_20230811.pdf.txt",
        default="../data/m&a/lumen/Lumen_Incumbent_Local_Exchange_Carrier_Business_Apollo_Global_Management_LLC_7_500m_Announce_20210803_merger_agree_20210804.pdf.txt",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Write the prompt to sql if not exist (temp here)
    # write_prompt_file_to_sql("credit.pass1.prompt.txt")
    # write_prompt_file_to_sql("credit.pass2.prompt.txt")

    # Extract the document text and inject tags
    document_text_with_tags = pdf2text_inject_tags(args.txtfile)
    filename = os.path.basename(args.txtfile)
    document_id = write_document_tags_to_sql(filename, document_text_with_tags)
    print(f"Document ID: {document_id}")

    tagged_text_process(
        document_id, "credit.pass1.prompt.txt", "credit.pass2.prompt.txt", args.limit
    )
