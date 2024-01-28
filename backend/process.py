import argparse
import os
import time
import concurrent.futures
from dotenv import load_dotenv
import openai
from openai import OpenAI
import json
from typing import List
import pandas as pd


# Load the OPENAI_API_KEY key from an environment file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No OpenAI API key found in environment variables")
openai_client = OpenAI(api_key=api_key)


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
    # replace newlines with spaces
    content = content.replace("\n", " ")
    return content.split("<PAGE ")


# Function to read and prepare the prompt
def prepare_prompt(page_text: str, prompt_path: str) -> str:
    with open(prompt_path, "r") as prompt_file:
        prompt = prompt_file.read()
    with open("resume_jd.txt", "r") as jd_file:
        jd = jd_file.read()
    prompt = prompt.replace("[resume_text]", page_text)
    prompt = prompt.replace("[job_description]", jd)
    return prompt


def send_to_chatgpt(prompt: str):
    t0 = time.time()
    try:
        response: openai.ChatCompletion = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"content": prompt, "role": "user"}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        j = json.loads(content)
        print(f"Processed in {time.time() - t0} seconds")
        return j
        # return ChatGPTResponse(**j, original_page_text=page_text)
    except openai.OpenAIError as e:
        # Handle any errors from the OpenAI API
        print(f"Error: {e} after {time.time() - t0} seconds")
        return {"error": str(e)}


# Main function to process the file and prompt
def process_file_and_prompt_multi_pages(
    txt_file: str, prompt_file: str
) -> pd.DataFrame:
    t0 = time.time()

    pages = read_and_split_file(txt_file)
    print(f"Read {len(pages)} pages from {txt_file} in {time.time() - t0} seconds")

    responses = []

    # Note PAGE_LIMIT is for testing purposes, so we don't have to wait for all pages to process before checking the results
    # PAGE_LIMIT = 4
    # for page in pages[:PAGE_LIMIT]:
    #     if page == "":
    #         continue
    #     response = send_to_chatgpt(page, prompt_file)
    #     handle_response(response)

    t1 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for page in pages:
            if page == "":
                continue
            future = executor.submit(send_to_chatgpt, page, prompt_file)
            futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            response = future.result()
            print(response)
            responses.append(response)

    print(f"Processed {len(pages)} pages in {time.time() - t1} seconds")
    return pd.DataFrame(responses)


def prepare_job_prompt(prompt: str, resume: str, job_description: str) -> str:
    prompt = prompt.replace("[resume_text]", resume)
    prompt = prompt.replace("[job_description]", job_description)
    return prompt


# Function to process the file and prompt
def process_file_and_prompt_single_send(
    resume_txt_file_path: str,
    prompt_txt_file_path: str,
    job_description_txt_file_path: str,
) -> pd.DataFrame:
    with open(resume_txt_file_path, "r") as r:
        resume = r.read()
    with open(job_description_txt_file_path, "r") as j:
        job_description = j.read()
    with open(prompt_txt_file_path, "r") as p:
        prompt = p.read()

    prompt = prepare_job_prompt(prompt, resume, job_description)

    response = send_to_chatgpt(prompt)
    with open(f"{resume_txt_file_path}.json", "w") as f:
        json.dump(response, f)
    return response


# Main function to process files in a directory and generate a single CSV output
def process_directory_and_prompt(
    directory_path: str, prompt_file: str, job_description_file: str
):
    all_responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                txt_file_path = os.path.join(directory_path, filename)
                print(f"Processing {txt_file_path}")
                future = executor.submit(
                    process_file_and_prompt_single_send,
                    txt_file_path,
                    prompt_file,
                    job_description_file,
                )
                futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            response = future.result()
            all_responses.append(response)
    # for filename in os.listdir(directory_path):
    #     if len(all_responses) > 3:
    #         break
    #     if filename.endswith(".txt"):
    #         txt_file_path = os.path.join(directory_path, filename)
    #         print(f"Processing {txt_file_path}")
    #         responses = process_file_and_prompt_single_send(
    #             txt_file_path, prompt_file, job_description_file
    #         )
    #         all_responses.append(responses)
    return all_responses


# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txtfile", type=str)  # , default="resume.txt"
    parser.add_argument("--dir", type=str, default="../outfolder")
    parser.add_argument("--prompt", default="resume_prompt.txt")
    parser.add_argument("--job_description", default="posting.txt")
    parser.add_argument("--csv", type=str, default="results.csv")
    args = parser.parse_args()

    # if args.csv:
    #     df = pd.read_csv(args.csv)

    if args.txtfile:
        resp = process_file_and_prompt_single_send(
            args.txtfile, args.prompt, args.job_description
        )
    elif args.dir:
        responses = process_directory_and_prompt(
            args.dir, args.prompt, args.job_description
        )
    else:
        raise ValueError("Must specify either --txtfile or --dir")

    # df.to_csv(args.csv, index=False)
