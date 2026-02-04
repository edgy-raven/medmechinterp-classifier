import argparse
import json
import os
import multiprocessing as mp
import numpy as np

from openai import OpenAI
from tqdm import tqdm

from utils import process_prompt_with_reasoning, extract_answer, hash_string_md5


SECOND_RESPONSE_TEMPLATE = (
    "Your reasoning patterns seem unusual. Approach the problem from a different angle. Format your response as previously."
)


def process_row_with_model(row, model_name, max_retries=5, retry_count=0):
    question = row['case_prompt']
    first_response = row['full_response']
    question_id = row['question_id']
    prompt = first_response + \
        SECOND_RESPONSE_TEMPLATE.format(case_prompt=question)
    response, reasoning = process_prompt_with_reasoning(prompt, model_name)
    if response is None and retry_count < max_retries:
        print(
            f"Retrying row {row.get('index', 'unknown')} for model {model_name} (attempt {retry_count + 1})")
        return process_row_with_model(row, model_name, retry_count=retry_count + 1)
    return {
        'original_message': {
            'role': 'user',
            'content': prompt,

        },
        'full_response': first_response + prompt + "\n<think>" + reasoning + "</think>\n" + response,
        'question_id': question_id,
        'first_response': first_response,
        'category': 'diagnosis',
        'question': row['case_prompt'],
        'gold_answer': row['gold_answer'],
        'extracted_answer': extract_answer(response),
        'reasoning_trace': reasoning,
        'pmcid': hash_string_md5(row['case_prompt']),
        'is_second_response': True,
    }


def process_chunk(args):
    chunk, model_name = args
    processed_chunk = []
    for entry in chunk:
        processed_chunk.append(process_row_with_model(entry, model_name))
    return processed_chunk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input JSON file')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name to use for generating second responses')
    parser.add_argument('-n', '--num_workers', type=int, default=4,
                        help='Number of parallel workers to use')
    args = parser.parse_args()

    with open(args.input, 'r') as infile:
        data = json.load(infile)

    chunks = np.array_split(data, args.num_workers)

    with mp.Pool(processes=args.num_workers) as pool:
        chunk_results = list(
            tqdm(pool.imap(process_chunk, [(chunk, args.model)
                 for chunk in chunks]), total=len(chunks))
        )

    results = [item for sublist in chunk_results for item in sublist]

    with open(args.input.split('.json')[0] + '.second.json', 'w') as outfile:
        json.dump(results, outfile, indent=2)
