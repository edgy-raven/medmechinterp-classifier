import json

import numpy as np
import pandas as pd

OUTPUT_FILENAME = "guided_accuracy.json"
PRED_COL = "predicted_judgement"


def inputs(context):
    labels_path = context.get("combined_features_path")
    if not labels_path:
        raise FileNotFoundError("features output missing; run with --jobs features first.")
    predictions_path = context.get("feature_xgb_predictions_path")
    if not predictions_path:
        raise FileNotFoundError("predictions missing; run with --jobs train first.")
    return {
        "labels_path": labels_path,
        "predictions_path": predictions_path,
        "output_dir": context["model_root"],
    }


def outputs(context):
    return {"guided_accuracy_path": context["model_root"] / OUTPUT_FILENAME}


def load_predictions(path, pred_col):
    preds = pd.read_csv(path)
    preds = preds[["pmcid", "model", pred_col]].copy()
    preds["model"] = preds["model"].astype(str)
    preds[pred_col] = preds[pred_col].astype(np.int32)
    return preds


def load_labels(path):
    labels = pd.read_csv(path)
    labels["model"] = labels["model"].astype(str)
    return labels


def compute_metrics(frame, pred_col):
    y_pred = frame[pred_col].to_numpy()
    y_first = frame["is_correct"].to_numpy()
    y_second = frame["is_second_correct"].to_numpy()
    model_col = frame["model"]

    metrics = {}
    model_names = ["overall"] + sorted(model_col.dropna().unique())
    for model_name in model_names:
        mask = np.ones(len(frame), dtype=bool) if model_name == "overall" else (model_col == model_name)
        y_pred_slice = y_pred[mask]
        y_first_slice = y_first[mask]
        y_second_slice = y_second[mask]
        guided = np.where(y_pred_slice == 0, y_second_slice, y_first_slice)
        metrics[model_name] = {
            "classifier_accuracy": float((y_pred_slice == y_first_slice).mean()),
            "first_accuracy": float(y_first_slice.mean()),
            "second_accuracy": float(y_second_slice.mean()),
            "guided_accuracy": float(np.mean(guided)),
            "oracle_guided_accuracy": float(np.maximum(y_first_slice, y_second_slice).mean()),
        }
    return metrics


def run(labels_path, predictions_path, output_dir, sources=None):
    labels = load_labels(labels_path)
    preds = load_predictions(predictions_path, PRED_COL)

    sources = sorted(labels["source"].dropna().unique().tolist())
    labels = labels[labels["source"].isin(sources)].copy()

    merged = labels.merge(preds, on=["pmcid", "model"], how="inner")
    merged = merged[merged["is_second_correct"].notna()].copy()
    if merged.empty:
        raise ValueError("No rows with second-response labels available for guided accuracy.")
    sources = sorted(merged["source"].dropna().unique().tolist())

    results = {}
    for source in sources:
        source_frame = merged[merged["source"] == source]
        results[source] = compute_metrics(source_frame, PRED_COL)

    output_path = output_dir / OUTPUT_FILENAME
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Saved guided accuracy metrics to {output_path}")
