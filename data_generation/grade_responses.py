import argparse
import json
import os
import multiprocessing as mp
import numpy as np

from openai import OpenAI
from tqdm import tqdm

from utils import process_openai_prompt


DESCRIBE_TEMPLATE = (
    "Here is a case presentation and the diagnosis. Describe only the provided diagnosis with respect to the case. Do not come up with your own diagnosis.\n\n"
    "----------------------------------------\n"
    "CASE\n"
    "----------------------------------------\n"
    "{case_prompt}\n\n"
    "----------------------------------------\n"
    "DIAGNOSIS\n"
    "----------------------------------------\n"
    "{diagnosis}"
)


COMPARE_TEMPLATE = (
    "Here is a case, a predicted diagnosis, and the true diagnosis.\n"
    "How similar are the two diagnoses? Answer only a number from 0-10.\n\n"
    "----------------------------------------\n"
    "CASE\n"
    "----------------------------------------\n"
    "{case_prompt}\n\n"
    "----------------------------------------\n\n"
    "PREDICTED DIAGNOSIS\n"
    "----------------------------------------\n"
    "{predicted_description}\n\n"
    "----------------------------------------\n"
    "TRUE DIAGNOSIS\n"
    "----------------------------------------\n"
    "{true_description}"
)


def grade_row(entry, client):
    pred = entry['extracted_answer']
    true = entry['gold_answer']
    case = entry['case_prompt']

    pred_description = process_openai_prompt(
        DESCRIBE_TEMPLATE.format(
            case_prompt=case,
            diagnosis=pred
        ),
        'gpt-5-nano',
        client,
    )
    true_description = process_openai_prompt(
        DESCRIBE_TEMPLATE.format(
            case_prompt=case,
            diagnosis=true
        ),
        'gpt-5-nano',
        client,
    )
    similarity_score = process_openai_prompt(
        COMPARE_TEMPLATE.format(
            case_prompt=case,
            predicted_description=pred_description,
            true_description=true_description,
        ),
        'gpt-5-nano',
        client,
    )

    if similarity_score:
        try:
            entry['similarity_score'] = float(similarity_score.strip())
            entry['_true_description'] = true_description
            entry['_predicted_description'] = pred_description
            if entry['similarity_score'] > 7:
                entry['is_correct'] = True
            else:
                entry['is_correct'] = False
        except ValueError:
            entry['similarity_score'] = None
            entry['is_correct'] = False

    return entry


def process_chunk(chunk):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    processed_chunk = []
    for entry in chunk:
        processed_chunk.append(grade_row(entry, client))
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
