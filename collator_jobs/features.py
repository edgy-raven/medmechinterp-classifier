import numpy as np
import pandas as pd

OUTPUT_FILENAME = "combined_features.csv"
CORRELATION_REPORT_LIMIT = 5
FEATURE_CORRELATION_REPORT_LIMIT = 10
CORRELATION_REPORT_THRESHOLD = 10


def inputs(context):
    return {
        "feature_paths": context["feature_context"].values(),
        "metadata_path": context["metadata_path"],
        "output_root": context["model_root"],
    }


def outputs(context):
    return {"combined_features_path": context["model_root"] / OUTPUT_FILENAME}


def run(feature_paths, metadata_path, output_root):
    metadata = pd.read_csv(metadata_path)[
        ["pmcid", "model", "source", "is_correct", "is_second_correct"]
    ]
    metadata["model"] = metadata["model"].astype(str)
    label_frame = metadata.set_index(["model", "pmcid"])["is_correct"]

    combined = metadata.copy()
    feature_frames = []
    for path in feature_paths:
        df = pd.read_csv(path)
        feature_frames.append((path, df))
        combined = combined.merge(df, on=["model", "pmcid"], how="left")

    for path, feature_df in feature_frames:
        numeric_cols = (
            feature_df.drop(columns=["pmcid", "model"], errors="ignore")
            .select_dtypes(include=[np.number])
            .columns.tolist()
        )
        numeric_cols = [name for name in numeric_cols if "emb" not in name.lower()]

        label_index = pd.MultiIndex.from_frame(feature_df[["model", "pmcid"]])
        labels = label_frame.reindex(label_index).to_numpy(dtype=float)

        correlations = []
        for name in numeric_cols:
            corr = float(np.corrcoef(feature_df[name].to_numpy(dtype=float), labels)[0, 1])
            correlations.append((abs(corr), corr, name))
            
        correlations.sort(reverse=True)
        if len(correlations) > CORRELATION_REPORT_THRESHOLD:
            print(f"Top {CORRELATION_REPORT_LIMIT} correlations with correctness for {path}:")
            for _, corr, name in correlations[:CORRELATION_REPORT_LIMIT]:
                print(f"  {name}: {corr:.3f}")
            print(f"Bottom {CORRELATION_REPORT_LIMIT} correlations with correctness for {path}:")
            for _, corr, name in correlations[-CORRELATION_REPORT_LIMIT:]:
                print(f"  {name}: {corr:.3f}")
        else:
            print(f"All correlations with correctness for {path}:")
            for _, corr, name in correlations:
                print(f"  {name}: {corr:.3f}")

    feature_df = combined.select_dtypes(include=[np.number]).drop(
        columns=["is_correct", "is_second_correct"], errors="ignore"
    )
    corr = feature_df.corr()
    upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    stacked = corr.where(upper).stack().reset_index()
    stacked.columns = ["feature_a", "feature_b", "corr"]
    stacked["abs_corr"] = stacked["corr"].abs()
    stacked = stacked.sort_values("abs_corr", ascending=False)
    top_k = min(FEATURE_CORRELATION_REPORT_LIMIT, len(stacked))
    print(f"Top {top_k} feature-feature correlations:")
    for _, row in stacked.head(top_k).iterrows():
        print(f"  {row['feature_a']} x {row['feature_b']}: {row['corr']:.3f}")

    collated_path = output_root / OUTPUT_FILENAME
    combined.to_csv(collated_path, index=False)
    print(f"Combined features written to {collated_path}")
