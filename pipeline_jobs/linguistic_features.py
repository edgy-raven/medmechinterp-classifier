from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import argparse

import numpy as np
import pandas as pd

import openai_helper

WORKERS = 48
ARG_PARSER = argparse.ArgumentParser(add_help=False)
ARG_PARSER.add_argument("--trace-embedding-dims", type=int, default=32)

SEEDS = {
    "reasoning_uncertainty": ['reasoning', 'suggest', 'maybe'],
    "disease": ['disease', 'abscess'],
    "neoplastic_mass_like": ['structure', 'cystic', 'tumor', 'mass', 'lesion'],
    "symptoms": ['injury', 'fever', 'pain', 'symptom', 'vomit'],
    "testing": ['ultrasound', 'biopsy'],
    "anatomy_organ_systems": ['lymph', 'renal', 'bone', 'liver', 'brain'],
    "clinical_course_management": ['treated', 'complication', 'woman', 'history', 'patient', 'cause', 'hospital']
}
SIM_THRESHOLDS = {
    "reasoning_uncertainty": 0.7,
    "disease": 0.4,
    "neoplastic_mass_like": 0.5,
    "symptoms": 0.6,
    "testing": 0.45,
    "anatomy_organ_systems": 0.45,
    "clinical_course_management": 0.45
}

OUTPUT_FILENAME = "linguistic_features.csv"


def inputs(context):
    token_sequences_path = context.get("token_sequences_path")
    if not token_sequences_path or not token_sequences_path.exists():
        raise FileNotFoundError("token sequences missing; run with --jobs token_sequence first.")
    parsed, _ = ARG_PARSER.parse_known_args(context["user_context"])
    return {
        "token_sequences_path": token_sequences_path,
        "output_dir": context["model_root"],
        "embedding_dims": parsed.trace_embedding_dims,
    }


def outputs(context):
    return {"linguistic_features_path": context["model_root"] / OUTPUT_FILENAME}


def compute_idf(token_sequences):
    doc_freq = Counter()
    for tokens in token_sequences:
        doc_freq.update(set(tokens))
    idf = {}
    for token, freq in doc_freq.items():
        idf[token] = np.log(len(token_sequences) / (1 + freq)) + 1
    return idf


def semantic_count(tokens, seeds, threshold, idf_map):
    total = 0.0
    for token in tokens:
        for seed in seeds:
            if openai_helper.similarity(token, seed) > threshold:
                total += idf_map[token]
    return total


def compute_tfidf_embedding(tokens, idf_map, dims):
    if not tokens:
        if dims and dims > 0:
            return [0.0 for _ in range(dims)]
        return []
    
    counts = Counter(tokens)
    unique_tokens = list(counts.keys())
    embeddings = openai_helper.embed_texts(unique_tokens)
    vec = np.zeros_like(np.array(embeddings[0], dtype=float))
    for token, emb in zip(unique_tokens, embeddings):
        weight = counts[token] * idf_map.get(token, 0.0)
        vec += weight * np.array(emb, dtype=float)

    if dims and dims > 0:
        reduced = vec[:dims]
        reduced_norm = np.linalg.norm(reduced)
        if reduced_norm:
            reduced = reduced / reduced_norm
        return reduced.tolist()
    vec_norm = np.linalg.norm(vec)
    if vec_norm:
        vec = vec / vec_norm
    return vec.tolist()


def compute_corr(token_sequences, seeds, threshold, idf_map, labels):
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        counts = np.array(list(
            executor.map(
                lambda seq: semantic_count(
                    seq,
                    seeds,
                    threshold,
                    idf_map
                ),
                token_sequences
            )
        ))
    if counts.std() == 0:
        return 0.0
    return abs(float(np.corrcoef(counts, labels)[0, 1]))


def extract_linguistic_features(pmcid, model, tokens, idf_map, embedding_dims):
    result = {"pmcid": pmcid, "model": model}
    for category in SEEDS:
        count = semantic_count(
            tokens,
            SEEDS[category],
            SIM_THRESHOLDS[category],
            idf_map
        )
        result[f"{category}_semantic_count"] = count

    tfidf_emb = compute_tfidf_embedding(tokens, idf_map, embedding_dims)
    for idx, value in enumerate(tfidf_emb):
        result[f"trace_emb_{idx}"] = value
    return result

def run(token_sequences_path, output_dir, embedding_dims):
    data = json.load(token_sequences_path.open())

    traces = data["traces"]
    token_sequences = [trace.get("token_sequence", []) for trace in traces]
    idf_map = compute_idf(token_sequences)

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        rows = list(
            executor.map(
                lambda trace: extract_linguistic_features(
                    trace.get("pmcid"),
                    trace.get("model"),
                    trace.get("token_sequence", []),
                    idf_map,
                    embedding_dims,
                ),
                traces,
            )
        )

    output_path = output_dir / OUTPUT_FILENAME
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved linguistic features to {output_path} ({len(rows)} rows)")
