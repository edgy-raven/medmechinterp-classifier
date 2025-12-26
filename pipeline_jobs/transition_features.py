import argparse
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import pandas as pd

OUTPUT_FILENAME = "transition_features.csv"
WORKERS = 48
ARG_PARSER = argparse.ArgumentParser(add_help=False)
ARG_PARSER.add_argument("--no-self-transition", action="store_true")
ARG_PARSER.add_argument("--normalize-transition-length", action="store_true")


def inputs(context):
    collated_path = context.get("annotator_path")
    if not collated_path or not collated_path.exists():
        raise FileNotFoundError("annotator output missing; run with --jobs annotator first.")
    metadata_path = context.get("metadata_path")
    if not metadata_path or not metadata_path.exists():
        raise FileNotFoundError("metadata output missing; run with --jobs metadata first.")
    parsed, _ = ARG_PARSER.parse_known_args(context["user_context"])
    return {
        "collated_path": collated_path,
        "metadata_path": metadata_path,
        "output_dir": context["model_root"],
        "no_self_transition": parsed.no_self_transition,
        "normalize_transition_length": parsed.normalize_transition_length,
    }


def outputs(context):
    return {"transition_features_path": context["model_root"] / OUTPUT_FILENAME}


def build_sequence(trace, no_self_transition):
    label_seq = trace.get("label_json") or []
    if not label_seq:
        return []
    seq = [label_seq[0]["taxonomy_tag"]]
    for entry in label_seq[1:]:
        tag = entry["taxonomy_tag"]
        if no_self_transition and tag == seq[-1]:
            continue
        seq.append(tag)
    return seq


def compute_transition_counts(seq, num_states):
    counts = np.zeros((num_states, num_states), dtype=int)
    for f, t in zip(seq, seq[1:]):
        counts[f][t] += 1
    return counts


def build_row(trace, seq, seq_counts, num_states, correct_probs, incorrect_probs, normalize_transition_length):
    row = {"pmcid": trace["pmcid"], "model": trace.get("model")}
    denom_counts = max(len(seq) - 1, 1) if normalize_transition_length else 1
    for f in range(num_states):
        for t in range(num_states):
            row[f"transition_{f}_{t}"] = float(seq_counts[f][t]) / denom_counts

    total_logprob_correct = 0.0
    total_logprob_incorrect = 0.0
    for f, t in zip(seq, seq[1:]):
        total_logprob_correct += float(np.log(correct_probs[f, t]))
        total_logprob_incorrect += float(np.log(incorrect_probs[f, t]))
    denom = max(len(seq) - 1, 1)
    row["avg_logprob_correct_matrix"] = total_logprob_correct / denom
    row["avg_logprob_incorrect_matrix"] = total_logprob_incorrect / denom
    row["avg_logprob_delta"] = row["avg_logprob_correct_matrix"] - row["avg_logprob_incorrect_matrix"]

    return row


def smooth_transition_probs(counts, epsilon):
    row_sums = counts.sum(axis=1, keepdims=True)
    return (counts + epsilon) / (row_sums + epsilon * counts.shape[1])


def run(
    collated_path,
    metadata_path,
    output_dir,
    no_self_transition,
    normalize_transition_length,
):
    data = json.load(collated_path.open())
    num_states = max(data["state_to_idx"].values()) + 1
    metadata = pd.read_csv(metadata_path)
    metadata["pmcid"] = metadata["pmcid"].astype(str)
    metadata["model"] = metadata["model"].astype(str)
    source_by_key = {
        (row["pmcid"], row["model"]): row["source"]
        for _, row in metadata.iterrows()
    }
    label_by_key = {
        (row["pmcid"], row["model"]): int(row["is_correct"])
        for _, row in metadata.iterrows()
    }

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        all_sequences = list(
            executor.map(lambda trace: build_sequence(trace, no_self_transition), data["traces"])
        )
        all_transition_counts = list(
            executor.map(lambda seq: compute_transition_counts(seq, num_states), all_sequences)
        )

    correct_counts = np.zeros((num_states, num_states), dtype=int)
    incorrect_counts = np.zeros((num_states, num_states), dtype=int)
    missing_label = 0
    for trace, seq_counts in zip(data["traces"], all_transition_counts):
        key = (str(trace["pmcid"]), str(trace.get("model")))
        source = source_by_key.get(key)
        if source is None:
            raise KeyError(
                f"Missing metadata source for pmcid={key[0]} model={key[1]}"
            )
        label = label_by_key.get(key)
        if label is None:
            missing_label += 1
            continue
        if source != "train":
            continue
        counts_target = correct_counts if label else incorrect_counts
        counts_target += seq_counts
    if missing_label:
        print(f"Skipping {missing_label} trace(s) without metadata labels for transition counts.")
    correct_probs = smooth_transition_probs(correct_counts, 1e-12)
    incorrect_probs = smooth_transition_probs(incorrect_counts, 1e-12)

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        rows = list(
            executor.map(
                build_row,
                data["traces"],
                all_sequences,
                all_transition_counts,
                repeat(num_states),
                repeat(correct_probs),
                repeat(incorrect_probs),
                repeat(normalize_transition_length),
            )
        )

    output_path = output_dir / OUTPUT_FILENAME
    df = pd.DataFrame(rows)

    if no_self_transition:
        constant_cols = [
            col
            for col in df.columns
            if col not in {"pmcid", "model"} and df[col].nunique(dropna=False) <= 1
        ]
        if constant_cols:
            df = df.drop(columns=constant_cols)

    df.to_csv(output_path, index=False)
    print(f"Wrote transition features to {output_path}")
