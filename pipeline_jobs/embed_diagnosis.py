import argparse
import json

import numpy as np
import pandas as pd
import openai_helper

OUTPUT_FILENAME = "predicted_diagnosis_embeddings.csv"
ARG_PARSER = argparse.ArgumentParser(add_help=False)
ARG_PARSER.add_argument("--predicted-diagnosis-embedding-dims", type=int, default=32)


def inputs(context):
    collated_path = context.get("annotator_path")
    if not collated_path or not collated_path.exists():
        raise FileNotFoundError("annotator output missing; run with --jobs annotator first.")
    parsed, _ = ARG_PARSER.parse_known_args(context["user_context"])
    return {
        "collated_path": collated_path,
        "output_dir": context["model_root"],
        "embedding_dims": parsed.predicted_diagnosis_embedding_dims,
    }


def outputs(context):
    return {"predicted_diagnosis_embeddings_path": context["model_root"] / OUTPUT_FILENAME}


def run(collated_path, output_dir, embedding_dims=None):
    data = json.load(collated_path.open())
    traces = data.get("traces", [])

    pred_texts = [t.get("predicted_diagnosis", "") for t in traces]
    pred_embs = openai_helper.embed_texts(pred_texts)
    if embedding_dims is not None:
        subsampled = []
        for emb in pred_embs:
            vec = np.array(emb, dtype=float)
            if embedding_dims > 0:
                vec = vec[:embedding_dims]
            norm = np.linalg.norm(vec)
            if norm:
                vec = vec / norm
            subsampled.append(vec.tolist())
        pred_embs = subsampled

    dim = len(pred_embs[0]) if pred_embs else 0
    pred_cols = [f"pred_diag_emb_{i}" for i in range(dim)]

    rows = []
    for trace, pe in zip(traces, pred_embs):
        row = {"pmcid": trace["pmcid"], "model": trace.get("model")}
        row.update({col: float(val) for col, val in zip(pred_cols, pe)})
        rows.append(row)

    output_path = output_dir / OUTPUT_FILENAME
    df = pd.DataFrame(rows)
    df = df.reindex(columns=["pmcid", "model"] + pred_cols)
    df.to_csv(output_path, index=False)
