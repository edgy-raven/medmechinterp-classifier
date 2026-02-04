import argparse
import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd

from datasets import load_dataset
from tqdm import tqdm

from grade_responses import grade_row
from utils import extract_answer, hash_string_md5, process_prompt_with_reasoning, MODELS

PROMPT_TEMPLATE = (
    "Read the following case presentation and give the most likely diagnosis.\n"
    "First, provide your internal reasoning for the diagnosis within the tags <think> ... </think>.\n"
    "Then, output the final diagnosis (just the name of the disease/entity) within the tags <answer> ... </answer>.\n\n"
    "----------------------------------------\nCASE PRESENTATION\n----------------------------------------\n"
    "{case_prompt}\n\n"
    "----------------------------------------\nOUTPUT TEMPLATE\n----------------------------------------\n"
    "<think>\n...your internal reasoning for the diagnosis...\n</think><answer>\n...the name of the disease/entity...\n</answer>"
)


def process_row_with_model(row, model_name, dataset_name, max_retries=5, retry_count=0):
    question = row['case_prompt']
    prompt = PROMPT_TEMPLATE.format(case_prompt=question)
    try:
        response, reasoning = process_prompt_with_reasoning(prompt, model_name)
        answer = extract_answer(response)
        if (answer == '' or response is None) and retry_count < max_retries:
            print(
                f"Retrying row {row.get('index', 'unknown')} for model {model_name} (attempt {retry_count + 1})")
            return process_row_with_model(row, model_name, dataset_name, retry_count=retry_count + 1)
        return {
            'original_message': {
                'role': 'user',
                'content': prompt,

            },
            'full_response': prompt + "\n<think>" + reasoning + "</think>\n" + response,
            'question_id': str(int(row['index'])) if row['index'] is str(row['index']).isnumeric() else row['index'],
            'category': 'diagnosis',
            'case_prompt': row['case_prompt'],
            'gold_answer': row['gold_answer'],
            'extracted_answer': answer,
            'dataset_name': dataset_name,
            'reasoning_trace': reasoning,
            'pmcid': hash_string_md5(row['case_prompt']),
        }
    except Exception as e:
        # Convert API exceptions to standard exceptions for pickling
        raise RuntimeError(
            f"Error processing row {row.get('index', 'unknown')} for model {model_name}: {str(e)}") from None


def process_subdataset_with_model(input_args):
    sub_df, model_name, dataset_name = input_args
    return_list = []
    for _, row in sub_df.iterrows():
        result = process_row_with_model(row, model_name, dataset_name)
        return_list.append(result)
    return pd.DataFrame(return_list)


def process_dataset_with_model(df, model_name, dataset_name, num_workers):
    divided_df = np.array_split(df, num_workers)
    with mp.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(pool.imap(process_subdataset_with_model, [(sub_df, model_name, dataset_name) for sub_df in divided_df]), total=len(divided_df)))
        new_df = pd.concat(results, axis=0, ignore_index=True)
    return new_df.to_dict(orient='records')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Dataset name to process')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to the input JSON file or HuggingFace dataset')
    parser.add_argument('-m', '--models', nargs='+', default=MODELS)
    parser.add_argument('-w', '--num_workers', type=int, default=4,
                        help='Number of parallel workers to use')
    parser.add_argument('-l', '--limit', type=int, default=None,
                        help='Limit number of rows to process')
    parser.add_argument('-o', '--output', type=str, default='new_artifacts',
                        help='Output folder to save results')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip processing if output file already exists')

    args = parser.parse_args()

    if args.dataset == 'medqa':
        df = pd.read_csv(args.path)
        df['case_prompt'] = df['question']
        df['gold_answer'] = df['answer']
    elif args.dataset == 'medmcqa':
        df = load_dataset(args.path,
                          split='train').to_pandas()
        df['index'] = df['id']
    elif args.dataset == 'medcasereasoning':
        df = load_dataset(args.path,
                          split='train').to_pandas()
        df['index'] = df['pmcid']
    elif args.dataset == 'nejm':
        df = pd.read_csv(args.path)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.limit is not None:
        df = df.head(args.limit)

    for model_name in args.models:
        os.makedirs(
            f"{args.output}/{model_name.split('/')[-1]}/{args.dataset}_with_second_responses", exist_ok=True)
        output_file = f"{args.output}/{model_name.split('/')[-1]}/{args.dataset}_with_second_responses/results_{model_name.split('/')[-1]}.json"
        if args.skip_existing and os.path.exists(output_file):
            print(
                f"Skipping model: {model_name} as output file already exists.")
            continue
        print(f"Processing model: {model_name} on dataset: {args.dataset}")
        results = process_dataset_with_model(
            df, model_name, args.dataset, num_workers=args.num_workers)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
