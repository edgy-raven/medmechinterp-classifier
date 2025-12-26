import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

OUTPUT_FILENAME = "metadata.csv"
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

ARG_PARSER = argparse.ArgumentParser(add_help=False)
ARG_PARSER.add_argument("--assign-splits", action="store_true")
ARG_PARSER.add_argument("--train-source", action="append", default=[])
ARG_PARSER.add_argument("--val-source", action="append", default=[])
ARG_PARSER.add_argument("--test-source", action="append", default=[])


def inputs(context):
    annotator_path = context.get("annotator_path")
    if not annotator_path or not annotator_path.exists():
        raise FileNotFoundError(
            "annotator output missing; run with --jobs annotator first."
        )

    second_response_path = context.get("second_response_path")
    if not second_response_path or not second_response_path.exists():
        raise FileNotFoundError(
            "second_response output missing; run with --jobs second_response first."
        )

    token_sequences_path = context.get("token_sequences_path")
    if token_sequences_path and not token_sequences_path.exists():
        token_sequences_path = None

    first_response_path = context.get("first_response_path")
    if not first_response_path or not first_response_path.exists():
        raise FileNotFoundError(
            "first_response output missing; run with --jobs first_response first."
        )

    parsed, _ = ARG_PARSER.parse_known_args(context["user_context"])
    return {
        "annotator_path": annotator_path,
        "output_root": context["output_root"],
        "first_response_path": first_response_path,
        "second_response_path": second_response_path,
        "token_sequences_path": token_sequences_path,
        "assign_splits": parsed.assign_splits,
        "train_sources": parsed.train_source,
        "val_sources": parsed.val_source,
        "test_sources": parsed.test_source,
    }


def outputs(context):
    return {"metadata_path": context["output_root"] / OUTPUT_FILENAME}


def _assign_splits(rows):
    train_split = TRAIN_SPLIT
    val_split = VAL_SPLIT
    test_split = TEST_SPLIT
    indices = np.arange(len(rows))
    
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1.0 - train_split),
        shuffle=True,
    )
    val_ratio = val_split / (val_split + test_split)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_ratio),
        shuffle=True,
    )
    for idx in train_idx:
        rows[int(idx)]["source"] = "train"
    for idx in val_idx:
        rows[int(idx)]["source"] = "val"
    for idx in test_idx:
        rows[int(idx)]["source"] = "test"


def _assign_train_val_splits(rows):
    train_split = TRAIN_SPLIT
    val_split = VAL_SPLIT
    total = train_split + val_split
    if total <= 0:
        raise ValueError("Train/val split fractions must sum to a positive value.")
    train_ratio = train_split / total
    n = len(rows)
    if n == 0:
        return
    indices = np.arange(n)
    if train_ratio <= 0:
        train_idx = np.array([], dtype=int)
        val_idx = indices
    elif train_ratio >= 1:
        train_idx = indices
        val_idx = np.array([], dtype=int)
    else:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=(1.0 - train_ratio),
            shuffle=True,
        )
    for idx in train_idx:
        rows[int(idx)]["source"] = "train"
    for idx in val_idx:
        rows[int(idx)]["source"] = "val"


def run(
    annotator_path,
    first_response_path,
    second_response_path,
    token_sequences_path,
    output_root,
    assign_splits=False,
    train_sources=None,
    val_sources=None,
    test_sources=None,
):
    annotator_data = json.load(annotator_path.open())
    traces = annotator_data["traces"]
    first_grades = {}
    for entry in json.load(first_response_path.open())["first_response_grades"]:
        pmcid = entry.get("pmcid")
        if not pmcid:
            continue
        first_grades[(entry["pmcid"], entry["model"])] = bool(entry.get("is_correct"))

    second_grades = {}
    for entry in json.load(second_response_path.open())["second_response_grades"]:
        pmcid = entry.get("pmcid")
        if not pmcid:
            continue
        second_grades[(entry["pmcid"], entry["model"])] = bool(entry.get("is_second_correct"))

    rows = []
    kept_traces = []
    missing_source = 0
    missing_first = 0
    missing_second = 0
    unmapped_sources = 0
    for trace in traces:
        pmcid = trace.get("pmcid")
        source = trace.get("source")
        if not pmcid or not source:
            missing_source += 1
            continue
        key = (pmcid, trace["model"])
        first_correct = first_grades.get(key)
        second_correct = second_grades.get(key)
        if first_correct is None or second_correct is None:
            if first_correct is None:
                missing_first += 1
            if second_correct is None:
                missing_second += 1
            continue
        rows.append({
            "pmcid": pmcid,
            "model": trace["model"],
            "source": source,
            "is_correct": int(bool(first_correct)),
            "is_second_correct": int(bool(second_correct)),
        })
        kept_traces.append(trace)

    train_sources = set(train_sources or [])
    val_sources = set(val_sources or [])
    test_sources = set(test_sources or [])
    if train_sources or val_sources or test_sources:
        mapped_rows = []
        mapped_traces = []
        train_rows = []
        train_traces = []
        seen_keys = set()
        for row, trace in zip(rows, kept_traces):
            src = row.get("source")
            if src in test_sources:
                row["source"] = "test"
                key = (row.get("pmcid"), row.get("model"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    mapped_rows.append(row)
                    mapped_traces.append(trace)
            elif src in val_sources:
                row["source"] = "val"
                key = (row.get("pmcid"), row.get("model"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    mapped_rows.append(row)
                    mapped_traces.append(trace)
            elif src in train_sources:
                key = (row.get("pmcid"), row.get("model"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    train_rows.append(row)
                    train_traces.append(trace)
            else:
                unmapped_sources += 1
        if train_rows:
            _assign_train_val_splits(train_rows)
            mapped_rows.extend(train_rows)
            mapped_traces.extend(train_traces)
        rows = mapped_rows
        kept_traces = mapped_traces
        if unmapped_sources:
            print(f"Skipped {unmapped_sources} trace(s) with unmapped sources.")
    else:
        sources = [row.get("source") for row in rows if row.get("source")]
        unique_sources = set(sources)
        required_sources = {"train", "val", "test"}
        should_assign = assign_splits or not required_sources.issubset(unique_sources)
        if should_assign:
            _assign_splits(rows)
        if rows:
            seen_keys = set()
            deduped_rows = []
            deduped_traces = []
            for row, trace in zip(rows, kept_traces):
                key = (row.get("pmcid"), row.get("model"))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                deduped_rows.append(row)
                deduped_traces.append(trace)
            rows = deduped_rows
            kept_traces = deduped_traces

    field_order = [
        "pmcid",
        "model",
        "source",
        "is_correct",
        "is_second_correct",
    ]
    df = pd.DataFrame(rows)
    ordered_cols = [col for col in field_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in ordered_cols]
    df = df[ordered_cols + remaining_cols]
    output_path = output_root / OUTPUT_FILENAME
    df.to_csv(output_path, index=False)
    should_rewrite = (
        missing_source
        or missing_first
        or missing_second
        or unmapped_sources
        or len(kept_traces) != len(traces)
    )
    if should_rewrite:
        annotator_data["traces"] = kept_traces
        with annotator_path.open("w") as f:
            json.dump(annotator_data, f, indent=2)
        print(f"Wrote filtered annotator output to {annotator_path} ({len(kept_traces)} traces).")
    if token_sequences_path:
        token_data = json.load(token_sequences_path.open())
        allowed_keys = {(row["pmcid"], row["model"]) for row in rows}
        seen = set()
        filtered_traces = []
        duplicate_keys = 0
        for trace in token_data.get("traces", []):
            key = (trace.get("pmcid"), trace.get("model"))
            if key not in allowed_keys:
                continue
            if key in seen:
                duplicate_keys += 1
                continue
            seen.add(key)
            filtered_traces.append(trace)
        token_data["traces"] = filtered_traces
        with token_sequences_path.open("w") as f:
            json.dump(token_data, f, indent=2)
        print(
            f"Wrote filtered token sequences to {token_sequences_path} ({len(filtered_traces)} traces)."
        )
        if duplicate_keys:
            print(f"Dropped {duplicate_keys} duplicate token sequence(s).")
    if missing_source:
        print(f"Skipped {missing_source} trace(s) without a source.")
    if missing_first:
        print(f"Skipped {missing_first} trace(s) without first response grades.")
    if missing_second:
        print(f"Skipped {missing_second} trace(s) without second responses.")
    print(f"Wrote metadata to {output_path} ({len(rows)} rows)")
