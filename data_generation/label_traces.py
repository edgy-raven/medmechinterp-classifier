import argparse
import json
import os
import multiprocessing as mp
import numpy as np
import ipdb

from openai import OpenAI
from tqdm import tqdm

from utils import process_openai_prompt, split_into_sentences

TAXONOMY = [
    # U1
    {"id": "orchestration_meta_control", "name": "Orchestration / meta-control"},
    {"id": "case_evidence_extraction",
        # U2
        "name": "Case evidence extraction (patient-specific facts)"},
    {"id": "evidence_processing",
        # U3
        "name": "Evidence processing (salience + interpretation + summarization)"},
    {"id": "medical_knowledge_general",
        # U4
        "name": "Medical knowledge / templates / criteria (general facts)"},
    {"id": "hypothesis_generation",
        # U5
        "name": "Hypothesis generation (differential expansion)"},
    {"id": "hypothesis_evaluation_narrowing",
        # U6
        "name": "Hypothesis evaluation & narrowing (support OR exclude)"},
    # U7
    {"id": "final_answer_commitment", "name": "Final answer commitment"},
]

# Universal taxonomy v2 state order (U1-U7)
STATE_ORDER = [
    "orchestration_meta_control",          # U1
    "case_evidence_extraction",           # U2
    "evidence_processing",                # U3
    "medical_knowledge_general",          # U4
    "hypothesis_generation",              # U5
    "hypothesis_evaluation_narrowing",    # U6
    "final_answer_commitment",            # U7
]
STATE_TO_IDX = {s: i for i, s in enumerate(STATE_ORDER)}

system_message = """You are an expert in interpreting how language models solve medical diagnostic cases using multi-step reasoning. Your task is to label sentences from a reasoning trace with taxonomy categories and identify dependencies between sentences.

Universal Taxonomy (U1-U7):

U1. Orchestration / meta-control
Task framing, stepwise strategy, attention directing ("now look at…"), self-correction/revision, clarifying prompt demand, output structuring.

U2. Case evidence extraction (patient-specific facts)
Any sentence that states vignette facts without explaining them: case presentation, symptoms, exam, labs as reported, imaging descriptions, timeline facts, PMH/FH/SH, meds, exposures, risk factors as facts.

U3. Evidence processing (salience + interpretation + summarization)
Flagging key clues/red flags, summarizing the presentation, selecting the focal problem, interpreting single data points (high/low/normal), interpreting demographics, and interpreting tempo/timing as compatibility judgments.

U4. Medical knowledge / templates / criteria (general facts)
Textbook recall: classic associations, typical presentations, diagnostic criteria listed/defined, etiologies, mechanistic facts stated as general knowledge.

U5. Hypothesis generation (differential expansion)
Proposing candidate diagnoses/etiologies/mechanisms for this case (including "what about…", reintroducing alternatives), without yet testing them.

U6. Hypothesis evaluation & narrowing (support OR exclude)
Any sentence that tests a hypothesis against evidence: multi-fact justification/synthesis, mechanism used to connect evidence→candidate, weighing competitors, ruling-out / "less likely because…".

U7. Final answer commitment
Explicit final diagnosis/etiology selection ("most likely is…", "therefore the answer is…").

other: Use only if the sentence does not fit any of the above tags or is purely stylistic.

For depends_on: Include a list of earlier sentence indices (0-based integers) that each sentence directly depends on. Only mark dependencies if the sentence clearly uses information, results, or logic from those earlier sentences. Use an empty list if there are no dependencies. Note: The indices will be converted to 1-based strings in the final output."""


def label_trace_with_tool_calling(
        client,
        case_prompt,
        reasoning_trace,
        model="gpt-5-nano",
        retries: int = 3):

    sentences = split_into_sentences(reasoning_trace)

    label_tool_definition = {
        "type": "function",
        "function": {
            "name": "label_sentences",
            "description": "Labels each sentence with appropriate taxonomy categories. Returns an array of sentence objects, each with 'index' (as string), 'text', and 'function' fields.",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": ["sentences"],
                "properties": {
                    "sentences": {
                        "type": "array",
                        "description": "Array of labeled sentence objects.",
                        "items": {
                            "type": "object",
                            "required": ["index", "text", "function"],
                            "properties": {
                                "index": {
                                    "type": "string",
                                    "description": "Sentence index (e.g., '0', '1', '2')."
                                },
                                "text": {
                                    "type": "string",
                                    "description": "The sentence text."
                                },
                                "function": {
                                    "type": "string",
                                    "enum": [
                                        "orchestration_meta_control",
                                        "case_evidence_extraction",
                                        "evidence_processing",
                                        "medical_knowledge_general",
                                        "hypothesis_generation",
                                        "hypothesis_evaluation_narrowing",
                                        "final_answer_commitment",
                                    ],
                                    "description": "Taxonomy category ID for this sentence."
                                }
                            },
                            "additionalProperties": False
                        }
                    }
                },
                "additionalProperties": False
            }
        }
    }

    user_message_label = f"""Medical case:
        {case_prompt}

        Sentences from reasoning trace (already split):
        {chr(10).join(f"{i}: {sent}" for i, sent in enumerate(sentences))}

        Please use the label_sentences tool to label each sentence with appropriate taxonomy categories. Return ALL {len(sentences)} sentences with their labels."""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message_label},
        ],
        tools=[label_tool_definition],
        tool_choice={"type": "function",
                     "function": {"name": "label_sentences"}},
    )

    message = json.loads(
        resp.choices[0].message.tool_calls[0].function.arguments)['sentences']

    return message


def label_row(entry, client):
    try:
        entry['label_json'] = label_trace_with_tool_calling(
            client,
            entry['case_prompt'],
            entry['reasoning_trace'],
            model="gpt-5-nano",
            retries=3,
        )
        entry['sequence'] = [STATE_TO_IDX[label['function']]
                             for label in entry['label_json']]
        entry['sequence_length'] = len(entry['sequence'])
    except Exception as e:
        print(f"Error labeling entry {entry.get('index', 'unknown')}: {e}")
        entry['label_json'] = None

    return entry


def process_chunk(chunk):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    processed_chunk = []
    for entry in chunk:
        processed_chunk.append(label_row(entry, client))
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
    final_results = {
        "state_order": STATE_ORDER,
        "state_to_idx": STATE_TO_IDX,
        "traces": results,
    }

    with open(f'{"/".join(args.input.split("/")[:-1])}/results.labeled.json', 'w') as outfile:
        json.dump(results, outfile, indent=2)
