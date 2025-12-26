import json
import re
from concurrent.futures import ThreadPoolExecutor

OUTPUT_FILENAME = "annotator_results.json"
WORKERS = 48


def inputs(context):
    return {
        "input_root": context["input_root"],
        "output_root": context["output_root"],
    }


def outputs(context):
    return {"annotator_path": context["output_root"] / OUTPUT_FILENAME}


def _process_trace(trace, state_to_idx, source_name, model_name):
    reasoning_text = trace["reasoning_trace"]
    if "<think" in reasoning_text.lower():
        reasoning_text = re.sub(r"</?think>", "", reasoning_text, flags=re.IGNORECASE).strip()

    label_json_raw = trace["label_json"]
    label_seq = []
    for key in sorted(label_json_raw.keys(), key=lambda k: int(k)):
        entry = label_json_raw[key]
        deps = []
        for dep in entry.get("depends_on") or []:
            dep_idx = int(dep)
            if dep_idx == int(key):
                continue
            deps.append(dep_idx - 1)
        label_seq.append({
            "taxonomy_tag": state_to_idx[entry["function"]],
            "depends_on": deps,
        })
    return {
        "reasoning_trace": reasoning_text,
        "pmcid": trace["pmcid"],
        "true_diagnosis": trace["true_diagnosis"],
        "predicted_diagnosis": trace["predicted_diagnosis"],
        "label_json": label_seq,
        "source": source_name.replace("_with_second_responses", ""),
        "model": model_name,
    }


def _iter_result_files(input_root):
    models = [d for d in input_root.iterdir() if d.is_dir()]
    for model_dir in models:
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            path = dataset_dir / "results.labeled.json"
            if path.exists():
                yield model_dir.name, path


def run(input_root, output_root):
    collated_path = output_root / OUTPUT_FILENAME

    combined_traces = []
    state_to_idx_global = None
    all_files = list(_iter_result_files(input_root))

    for model_name, path in all_files:
        data = json.load(path.open())
        if state_to_idx_global is None:
            state_to_idx_global = data["state_to_idx"]

        source_name = path.parent.name
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            combined_traces.extend(
                executor.map(
                    lambda tr: _process_trace(tr, state_to_idx_global, source_name, model_name),
                    data["traces"],
                )
            )
    collated_payload = {
        "traces": combined_traces,
        "state_to_idx": state_to_idx_global,
    }
    with collated_path.open("w") as f:
        json.dump(collated_payload, f, indent=2)

    print(f"Collated {len(all_files)} file(s) into {collated_path} with {len(combined_traces)} traces")
