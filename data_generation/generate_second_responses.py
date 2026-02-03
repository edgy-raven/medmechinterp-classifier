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


def process_row_with_model(row, model_name, dataset_name, max_retries=5, retry_count=0):
    question = row['case_prompt']
    first_response = row['full_response']
    prompt = first_response + \
        SECOND_RESPONSE_TEMPLATE.format(case_prompt=question)
    try:
        response, reasoning = process_prompt_with_reasoning(prompt, model_name)
        if response is None and retry_count < max_retries:
            print(
                f"Retrying row {row.get('index', 'unknown')} for model {model_name} (attempt {retry_count + 1})")
            return process_row_with_model(row, model_name, dataset_name, retry_count=retry_count + 1)
        return {
            'original_message': {
                'role': 'user',
                'content': prompt,

            },
            'full_response': first_response + prompt + "\n<think>" + reasoning + "</think>\n" + response,
            'question_id': str(int(row['index'])) if row['index'] is str(row['index']).isnumeric() else row['index'],
            'first_response': first_response,
            'category': 'diagnosis',
            'question': row['case_prompt'],
            'gold_answer': row['gold_answer'],
            'extracted_answer': extract_answer(response),
            'dataset_name': dataset_name,
            'reasoning_trace': reasoning,
            'pmcid': hash_string_md5(row['case_prompt']),
            'is_second_response': True,
        }
    except Exception as e:
        # Convert API exceptions to standard exceptions for pickling
        raise RuntimeError(
            f"Error processing row {row.get('index', 'unknown')} for model {model_name}: {str(e)}") from None


def process_chunk(chunk):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    processed_chunk = []
    for entry in chunk:
        processed_chunk.append(process_row_with_model(entry, client))
    return processed_chunk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input JSON file')
    parser.add_argument('-n', '--num_workers', type=int, default=4,
                        help='Number of parallel workers to use')
    args = parser.parse_args()

    with open(args.input, 'r') as infile:
        data = json.load(infile)

    chunks = np.array_split(data, args.num_workers)

    with mp.Pool(processes=args.num_workers) as pool:
        chunk_results = list(
            tqdm(pool.imap(process_chunk, chunks), total=len(chunks))
        )

    results = [item for sublist in chunk_results for item in sublist]

    with open(args.input.split('.json')[0] + '.graded.json', 'w') as outfile:
        json.dump(results, outfile, indent=2)
