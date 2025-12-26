import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

OUTPUT_FILENAME = "dependency_features.csv"
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
    return {"dependency_features_path": context["model_root"] / OUTPUT_FILENAME}


def find_roots(deps):
    indegree = [0 for _ in range(len(deps))]
    for entry in deps:
        for dep in entry:
            if 0 <= dep < len(indegree):
                indegree[dep] += 1

    return [idx for idx, deg in enumerate(indegree) if deg == 0]


def extract_dependency_features(trace):
    label_seq = trace.get("label_json") or []
    deps = []
    for label in label_seq:
        raw_deps = label.get("depends_on")
        if raw_deps is None:
            raw_deps = []
        elif not isinstance(raw_deps, list):
            raw_deps = [raw_deps]
        normalized = []
        for dep in raw_deps:
            try:
                dep_idx = int(dep)
            except (TypeError, ValueError):
                continue
            normalized.append(dep_idx)
        deps.append(normalized)
    roots = find_roots(deps)
    return {
        "pmcid": trace["pmcid"],
        "model": trace.get("model"),
        "root_count": len(roots),
    }


def run(annotator_path, output_dir):
    data = json.load(annotator_path.open())
    traces = data["traces"]

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        rows = list(executor.map(extract_dependency_features, traces))

    output_path = output_dir / OUTPUT_FILENAME
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved dependency features to {output_path} ({len(rows)} rows)")
