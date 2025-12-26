import json
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

OUTPUT_FILENAME = "simple_text_features.csv"
WORKERS = 48


def inputs(context):
    annotator_path = context.get("annotator_path")
    if not annotator_path or not annotator_path.exists():
        raise FileNotFoundError("annotator output missing; run with --jobs annotator first.")

    return {
        "annotator_path": annotator_path,
        "output_dir": context["model_root"],
    }


def outputs(context):
    return {"simple_text_features_path": context["model_root"] / OUTPUT_FILENAME}


def extract_simple_text_features(trace,):
    text = trace["reasoning_trace"] or ""
    words = text.split()
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    avg_sentence_len = word_count / len(sentences) if sentences else 0
    return {
        "pmcid": trace["pmcid"],
        "model": trace.get("model"),
        "text_length": len(text),
        "sentence_length": avg_sentence_len,
        "question_marks": text.count("?"),
        "capitals": len(re.findall(r"\b[A-Z]{2,}\b", text)),
    }


def run(annotator_path, output_dir):
    data = json.load(annotator_path.open())
    traces = data["traces"]

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        rows = list(
            executor.map(
                lambda trace: extract_simple_text_features(trace),
                traces,
            )
        )

    output_path = output_dir / OUTPUT_FILENAME
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved simple text features to {output_path} ({len(rows)} rows)")
