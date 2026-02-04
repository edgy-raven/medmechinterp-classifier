import anthropic
import json
import pandas as pd
import time
import ipdb

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from openai import OpenAI
from google.genai import types
from google import genai

from utils import extract_answer, hash_string_md5

PROMPT_TEMPLATE = (
    "Read the following case presentation and give the most likely diagnosis.\n"
    "First, provide your internal reasoning for the diagnosis within the tags <think> ... </think>.\n"
    "Then, output the final diagnosis (just the name of the disease/entity) within the tags <answer> ... </answer>.\n\n"
    "----------------------------------------\nCASE PRESENTATION\n----------------------------------------\n"
    "{}\n\n"
    "----------------------------------------\nOUTPUT TEMPLATE\n----------------------------------------\n"
    "<think>\n...your internal reasoning for the diagnosis...\n</think><answer>\n...the name of the disease/entity...\n</answer>"
)

data = pd.read_csv(
    '../../MedQA_complete_graded_data.csv').sample(n=100, random_state=42)
data['pmcid'] = data['question'].apply(hash_string_md5)

# Step 1: Client initialization for all clients
anthropic_client = anthropic.Anthropic()
gemini_client = genai.Client()
openai_client = OpenAI()

# Step 2: Batch creation for all models
anthropic_message_batch = anthropic_client.messages.batches.create(
    requests=[
        Request(
            custom_id=case['pmcid'],
            params=MessageCreateParamsNonStreaming(
                model="claude-sonnet-4-5",
                messages=[{
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(case['question']),
                }],
                max_tokens=16384
            )
        ) for case in data.to_dict(orient='records')
    ]
)

# Create a sample JSONL file
with open("gemini-batch-requests.jsonl", "w") as f:
    requests = [
        {
            "key": case['pmcid'],
            "request": {
                "contents": [
                    {
                        "parts": [{"text": PROMPT_TEMPLATE.format(case['question'])}]
                    }
                ]
            }
        } for case in data.to_dict(orient='records')
    ]
    for req in requests:
        f.write(json.dumps(req) + "\n")

with open("openbatch-requests.jsonl", "w") as f:
    requests = [
        {
            "custom_id": case['pmcid'],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-5",
                "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(case['question'])}],
            }
        }
        for case in data.to_dict(orient='records')]
    for req in requests:
        f.write(json.dumps(req) + "\n")


# Step 3: Begin batch jobs for all models
openai_input_file = openai_client.files.create(
    file=open("openbatch-requests.jsonl", "rb"),
    purpose="batch"
)
openai_batch_job = openai_client.batches.create(
    input_file_id=openai_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "nightly eval job"
    }
)

openai_batch_job_id = openai_batch_job.id
openai_batch_job = openai_client.batches.retrieve(openai_batch_job_id)

gemini_input_file = gemini_client.files.upload(
    file='gemini-batch-requests.jsonl',
    config=types.UploadFileConfig(
        display_name='gemini-batch-requests', mime_type='jsonl')
)
gemini_batch_job = gemini_client.batches.create(
    model="gemini-3-pro-preview",
    src=gemini_input_file.name,
    config={
        'display_name': "file-upload-job-1",
    },
)

gemini_job_name = gemini_batch_job.name
gemini_batch_job = gemini_client.batches.get(name=gemini_job_name)

completed_states = set([
    'JOB_STATE_SUCCEEDED',
    'JOB_STATE_FAILED',
    'JOB_STATE_CANCELLED',
    'JOB_STATE_EXPIRED',
])

gemini_batch_job = gemini_client.batches.get(
    name=gemini_job_name)  # Initial get

# Step 4: Monitoring batch jobs for all models
gemini_job_done = False
gemini_results_processed = False
gemini_results = None

anthropic_job_done = False
anthropic_results_processed = False
anthropic_results = None

openai_job_done = False
openai_results_processed = False
openai_results = None

start_time = time.time()

while not gemini_job_done or not anthropic_job_done or not openai_job_done:
    openai_batch_job = openai_client.batches.retrieve(openai_batch_job_id)
    openai_job_done = openai_batch_job.status in [
        "failed", "completed", "expired", "cancelled"]
    anthropic_message_batch = anthropic_client.messages.batches.retrieve(
        anthropic_message_batch.id
    )
    anthropic_job_done = anthropic_message_batch.processing_status == "ended"
    gemini_batch_job = gemini_client.batches.get(name=gemini_job_name)
    gemini_job_done = gemini_batch_job.state.name in completed_states
    if not gemini_job_done:
        print(
            f"Gemini batch is still processing (job state: {gemini_batch_job.state.name})...")
    else:
        print(
            f"Gemini batch job finished with state: {gemini_batch_job.state.name}")
        if not gemini_results_processed:
            gemini_results_processed = True
            print(
                f"Gemini job finished with state: {gemini_batch_job.state.name}")
            if gemini_batch_job.state.name == 'JOB_STATE_FAILED':
                print(f"Error: {gemini_batch_job.error}")
            gemini_batch_job = gemini_client.batches.get(name=gemini_job_name)

            if gemini_batch_job.state.name == 'JOB_STATE_SUCCEEDED':

                # If batch job was created with a file
                if gemini_batch_job.dest and gemini_batch_job.dest.file_name:
                    # Results are in a file
                    result_file_name = gemini_batch_job.dest.file_name
                    print(f"Results are in file: {result_file_name}")

                    print("Downloading result file content...")
                    file_content = gemini_client.files.download(
                        file=result_file_name)
                    # Process file_content (bytes) as needed
                    gemini_results = pd.read_json(
                        file_content.decode('utf-8'), lines=True)

                    gemini_results = [{'pmcid': x['key'], 'text': x['response']['candidates'][0]['content']['parts'][0]['text'], 'input_tokens': x['response']['usageMetadata']['promptTokenCount'],
                                       'output_tokens': x['response']['usageMetadata']['candidatesTokenCount']+x['response']['usageMetadata']['thoughtsTokenCount'], 'extracted_answer': extract_answer(x['response']['candidates'][0]['content']['parts'][0]['text'])} for x in gemini_results.to_dict(orient='records')]

                    for res in gemini_results:
                        res['case_prompt'] = data[data['pmcid']
                                                  == res['pmcid']]['question'].values[0]
                        res['gold_answer'] = data[data['pmcid']
                                                  == res['pmcid']]['answer'].values[0]
                        res['question_id'] = data[data['pmcid']
                                                  == res['pmcid']]['index'].values[0]

    if not openai_job_done:
        print(
            f"OpenAI batch is still processing (job state: {openai_batch_job.status})...")
    else:
        print(
            f"OpenAI batch job finished with state: {openai_batch_job.status}")
        if not openai_results_processed:
            openai_results_processed = True
            openai_batch_job = openai_client.batches.retrieve(
                openai_batch_job_id)
            openai_results = openai_client.files.content(
                openai_batch_job.output_file_id).text
            openai_results = pd.read_json(
                openai_results, lines=True)
            openai_results = [{'pmcid': x['custom_id'], 'text': x['response']['body']['choices'][0]['message']['content'], 'input_tokens': x['response']['body']['usage']['prompt_tokens'],
                              'output_tokens': x['response']['body']['usage']['completion_tokens'], 'extracted_answer': extract_answer(x['response']['body']['choices'][0]['message']['content'])} for x in openai_results.to_dict(orient='records')]
            for res in openai_results:
                res['case_prompt'] = data[data['pmcid']
                                          == res['pmcid']]['question'].values[0]
                res['gold_answer'] = data[data['pmcid']
                                          == res['pmcid']]['answer'].values[0]
                res['question_id'] = data[data['pmcid']
                                          == res['pmcid']]['index'].values[0]

    if not anthropic_job_done:
        print(
            f"Anthropic batch is still processing (job state: {anthropic_message_batch.processing_status})...")
    else:
        print(
            f"Anthropic batch job finished with state: {anthropic_message_batch.processing_status}")
        if not anthropic_results_processed:
            anthropic_results_processed = True
            message_batch_individual_response = anthropic_client.messages.batches.results(
                anthropic_message_batch.id,
            )
            anthropic_results = []
            for res in message_batch_individual_response:
                case_prompt = data[data['pmcid'] ==
                                   res.custom_id]['question'].values[0]
                answer = data[data['pmcid'] ==
                              res.custom_id]['answer'].values[0]
                index = data[data['pmcid'] == res.custom_id]['index'].values[0]
                anthropic_results.append({
                    'pmcid': res.custom_id,
                    'question_id': index,
                    'text': res.result.message.content[0].text,
                    'case_prompt': case_prompt,
                    'gold_answer': answer,
                    'extracted_answer': extract_answer(res.result.message.content[0].text),
                    'input_tokens': res.result.message.usage.input_tokens,
                    'output_tokens': res.result.message.usage.output_tokens
                })

    time.sleep(60)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {int(elapsed_time//60)} minute(s)")

with open("proprietary_models_artifacts/gemini_results.json", "w") as f:
    json.dump(gemini_results, f, indent=2)
with open("proprietary_models_artifacts/anthropic_results.json", "w") as f:
    json.dump(anthropic_results, f, indent=2)
with open("proprietary_models_artifacts/openai_results.json", "w") as f:
    json.dump(openai_results, f, indent=2)
